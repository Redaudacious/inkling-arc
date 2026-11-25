// inkconv: rebuild an Inkling 2 Mini checkpoint in the B70 serving format from a BF16 fine-tune (Claude, 24 Nov 2026).
// Made for DavidAU's Inkling-2-Mini-26B-Heretic-...-50X-V2-AGI (BF16, 52.8 GB): the inkling-arc engine reads the
// thinkingmachines checkpoint layout, so the fine-tune is put into that layout.
//
// The base checkpoint is the template: same tensors, order, dtypes and scales.
//   experts, layers 0-19  NVFP4: two E2M1 codes per byte (low nibble first), an E4M3 scale per 16 values and a per-expert
//                         factor amax / outer_scale_denominator
//   experts, layers 20-23 MXFP8: E4M3 values and a 2^(s-127) scale per 32 values
// Each expert of the fine-tune is quantized onto the base's own scales (nearest code), so an expert the fine-tune left
// unchanged comes out byte-identical to the base. A block whose values no longer fit its scale gets a larger scale; an
// NVFP4 expert that outgrows its factor gets a new factor and fresh scales. Every other tensor is copied from the
// fine-tune in the base's dtype. Rust std only.
//   inkconv <base dir> <fine-tune dir> <out dir> [check]      check = statistics only, nothing written
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::time::Instant;

// ---------- minimal JSON (safetensors headers only) ----------
#[derive(Clone, Debug)]
enum J { Null, Bool(bool), Num(String), Str(String), Arr(Vec<J>), Obj(Vec<(String, J)>) }

struct P<'a> { s: &'a [u8], i: usize }
impl<'a> P<'a> {
    fn ws(&mut self) { while self.i < self.s.len() && (self.s[self.i] as char).is_whitespace() { self.i += 1; } }
    fn val(&mut self) -> J {
        self.ws();
        match self.s[self.i] {
            b'{' => { self.i += 1; let mut o = vec![]; loop { self.ws(); if self.s[self.i] == b'}' { self.i += 1; break; }
                let k = match self.val() { J::Str(k) => k, _ => panic!("key") }; self.ws(); assert_eq!(self.s[self.i], b':'); self.i += 1;
                let v = self.val(); o.push((k, v)); self.ws(); if self.s[self.i] == b',' { self.i += 1; } } J::Obj(o) }
            b'[' => { self.i += 1; let mut a = vec![]; loop { self.ws(); if self.s[self.i] == b']' { self.i += 1; break; }
                a.push(self.val()); self.ws(); if self.s[self.i] == b',' { self.i += 1; } } J::Arr(a) }
            b'"' => { self.i += 1; let mut out = String::new(); loop { let c = self.s[self.i]; self.i += 1;
                match c { b'"' => break, b'\\' => { let e = self.s[self.i]; self.i += 1; match e {
                    b'n' => out.push('\n'), b't' => out.push('\t'), b'r' => out.push('\r'), b'b' => out.push('\u{8}'), b'f' => out.push('\u{c}'),
                    b'u' => { let h = std::str::from_utf8(&self.s[self.i..self.i + 4]).unwrap(); self.i += 4;
                              out.push(char::from_u32(u32::from_str_radix(h, 16).unwrap()).unwrap_or('?')); }
                    o => out.push(o as char) } }
                  _ => { let st = self.i - 1; let mut en = self.i; while self.s[en] != b'"' && self.s[en] != b'\\' { en += 1; }
                         out.push_str(std::str::from_utf8(&self.s[st..en]).unwrap()); self.i = en; } } } J::Str(out) }
            b't' => { self.i += 4; J::Bool(true) }
            b'f' => { self.i += 5; J::Bool(false) }
            b'n' => { self.i += 4; J::Null }
            _ => { let st = self.i; while self.i < self.s.len() && b"+-0123456789.eE".contains(&self.s[self.i]) { self.i += 1; }
                   J::Num(std::str::from_utf8(&self.s[st..self.i]).unwrap().to_string()) }
        }
    }
}
fn esc(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() { match c { '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"),
        c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)), c => o.push(c) } }
    o
}
fn dump(j: &J) -> String {
    match j { J::Null => "null".into(), J::Bool(b) => b.to_string(), J::Num(n) => n.clone(), J::Str(s) => format!("\"{}\"", esc(s)),
        J::Arr(a) => format!("[{}]", a.iter().map(dump).collect::<Vec<_>>().join(",")),
        J::Obj(o) => format!("{{{}}}", o.iter().map(|(k, v)| format!("\"{}\":{}", esc(k), dump(v))).collect::<Vec<_>>().join(",")) }
}

// ---------- safetensors ----------
#[derive(Clone)]
struct T { name: String, dtype: String, shape: Vec<usize>, a: u64, b: u64 }
struct St { f: File, base: u64, ts: Vec<T>, by: HashMap<String, usize>, meta: Option<J> }
impl St {
    fn open(path: &str) -> St {
        let mut f = File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));
        let mut l = [0u8; 8]; f.read_exact(&mut l).unwrap();
        let n = u64::from_le_bytes(l); let mut h = vec![0u8; n as usize]; f.read_exact(&mut h).unwrap();
        let mut p = P { s: &h, i: 0 };
        let (mut ts, mut meta) = (vec![], None);
        if let J::Obj(o) = p.val() { for (k, v) in o {
            if k == "__metadata__" { meta = Some(v); continue; }
            let J::Obj(f) = v else { panic!("tensor {}", k) };
            let g = |n: &str| f.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone()).unwrap();
            let nums = |v: J| -> Vec<u64> { match v { J::Arr(a) => a.iter().map(|x| match x { J::Num(n) => n.parse().unwrap(), _ => panic!() }).collect(), _ => panic!() } };
            let dtype = match g("dtype") { J::Str(s) => s, _ => panic!() };
            let off = nums(g("data_offsets"));
            ts.push(T { name: k, dtype, shape: nums(g("shape")).iter().map(|&x| x as usize).collect(), a: off[0], b: off[1] });
        } }
        ts.sort_by_key(|t| t.a);
        let by = ts.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
        St { f, base: 8 + n, ts, by, meta }
    }
    fn get(&self, n: &str) -> Option<&T> { self.by.get(n).map(|&i| &self.ts[i]) }
    fn bytes(&mut self, n: &str) -> Vec<u8> {
        let t = self.get(n).unwrap_or_else(|| panic!("missing tensor {}", n)).clone();
        let mut v = vec![0u8; (t.b - t.a) as usize];
        self.f.seek(SeekFrom::Start(self.base + t.a)).unwrap(); self.f.read_exact(&mut v).unwrap(); v
    }
}

// ---------- number formats ----------
fn e4m3_dec(b: u8) -> f32 {
    let (s, e, m) = (b >> 7, ((b >> 3) & 15) as i32, (b & 7) as f32);
    if e == 15 && (b & 7) == 7 { return f32::NAN; }
    let v = if e == 0 { m / 8.0 * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) };
    if s == 1 { -v } else { v }
}
struct E4 { pos: Vec<f32> } // values of codes 0..=126 (ascending)
impl E4 {
    fn new() -> E4 { E4 { pos: (0u8..=126).map(e4m3_dec).collect() } }
    fn enc(&self, x: f32) -> u8 { // nearest, ties to even code, saturates at 448
        let (s, a) = (if x < 0.0 { 0x80u8 } else { 0 }, x.abs());
        if !(a < self.pos[126]) { return s | 126; }
        let i = self.pos.partition_point(|&v| v <= a) - 1; // pos[i] <= a < pos[i+1]
        let (lo, hi) = (a - self.pos[i], self.pos[i + 1] - a);
        let c = if lo < hi { i } else if hi < lo { i + 1 } else if i % 2 == 0 { i } else { i + 1 };
        s | c as u8
    }
    fn up(&self, x: f32) -> Option<u8> { // smallest positive code with value >= x
        if x > self.pos[126] { return None; }
        Some(self.pos.partition_point(|&v| v < x).max(1) as u8)
    }
}
const E2: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
fn e2_enc(x: f32) -> u8 { // nearest E2M1 code (sign in bit 3), ties to even magnitude index
    let a = x.abs();
    let mut i = 0usize;
    while i < 7 && a > E2[i + 1] { i += 1; }
    let c = if i == 7 { 7 } else { let (lo, hi) = (a - E2[i], E2[i + 1] - a);
        if lo < hi { i } else if hi < lo { i + 1 } else if i % 2 == 0 { i } else { i + 1 } };
    (if x < 0.0 && c != 0 { 8 } else { 0 }) | c as u8
}
fn bf(src: &[u8], i: usize) -> f32 { f32::from_bits((u16::from_le_bytes([src[2 * i], src[2 * i + 1]]) as u32) << 16) }
fn to_bf16(x: f32) -> [u8; 2] { let u = x.to_bits(); let r = (u + 0x7fff + ((u >> 16) & 1)) >> 16; (r as u16).to_le_bytes() }

#[derive(Default, Clone, Copy)]
struct Stat { experts_changed: usize, codes_changed: u64, blocks_rescaled: u64, experts_refactored: usize, max_abs_dev: f32 }
impl Stat { fn add(&mut self, o: Stat) { self.experts_changed += o.experts_changed; self.codes_changed += o.codes_changed;
    self.blocks_rescaled += o.blocks_rescaled; self.experts_refactored += o.experts_refactored; self.max_abs_dev = self.max_abs_dev.max(o.max_abs_dev); } }

// rows of the fine-tune for checkpoint row r (w13 may be stored interleaved or as [gate; up])
fn src_row(r: usize, rows: usize, deint: bool) -> usize { if !deint { r } else if r % 2 == 0 { r / 2 } else { rows / 2 + r / 2 } }

// NVFP4 expert: codes (R, C/2), scales (R, C/16), factor; src bf16 (R, C). Returns stats; rewrites codes/scales/amax in place.
fn nv_expert(e4: &E4, codes: &mut [u8], scales: &mut [u8], amax: &mut f32, den: f32, src: &[u8], rows: usize, cols: usize, deint: bool) -> Stat {
    let mut st = Stat::default();
    let w = |r: usize, c: usize| bf(src, src_row(r, rows, deint) * cols + c);
    // the fine-tune's largest value decides whether the base factor still holds (scales top out at 448)
    let mut emax = 0f32; for r in 0..rows { for c in 0..cols { emax = emax.max(w(r, c).abs()); } }
    let mut fac = *amax / den;
    // tolerances cover bf16 rounding of a dequantized base value (relative 2^-9) so unchanged experts never move
    if emax / 6.0 / fac > 448.0 * 1.004 { *amax = emax; fac = emax / den; st.experts_refactored = 1;
        for r in 0..rows { for b in 0..cols / 16 { let mut m = 0f32; for j in 0..16 { m = m.max(w(r, b * 16 + j).abs()); }
            scales[r * cols / 16 + b] = e4.up(m / 6.0 / fac).unwrap_or(126); } } }
    for r in 0..rows { for b in 0..cols / 16 {
        let si = r * cols / 16 + b;
        let mut m = 0f32; for j in 0..16 { m = m.max(w(r, b * 16 + j).abs()); }
        let mut unit = e4m3_dec(scales[si]) * fac;
        if m > 6.0 * unit * 1.004 || (unit == 0.0 && m > 0.0) { scales[si] = e4.up(m / 6.0 / fac).unwrap_or(126); unit = e4m3_dec(scales[si]) * fac; st.blocks_rescaled += 1; }
        for j in 0..16 { let c = b * 16 + j; let x = w(r, c);
            let q = if unit > 0.0 { e2_enc(x / unit) } else { 0 };
            let bi = r * cols / 2 + c / 2; let old = if c % 2 == 0 { codes[bi] & 15 } else { codes[bi] >> 4 };
            let deq = { let v = E2[(old & 7) as usize] * unit; if old & 8 != 0 { -v } else { v } };
            st.max_abs_dev = st.max_abs_dev.max((deq - x).abs());
            if q != old { st.codes_changed += 1; codes[bi] = if c % 2 == 0 { (codes[bi] & 0xF0) | q } else { (codes[bi] & 0x0F) | (q << 4) }; } }
    } }
    if st.codes_changed > 0 || st.blocks_rescaled > 0 || st.experts_refactored > 0 { st.experts_changed = 1; }
    st
}

// MXFP8 expert: vals (R, C) e4m3, scales (R, C/32) u8 exponent; src bf16 (R, C)
fn mx_expert(e4: &E4, vals: &mut [u8], scales: &mut [u8], src: &[u8], rows: usize, cols: usize, deint: bool) -> Stat {
    let mut st = Stat::default();
    let w = |r: usize, c: usize| bf(src, src_row(r, rows, deint) * cols + c);
    for r in 0..rows { for b in 0..cols / 32 {
        let si = r * cols / 32 + b;
        let mut m = 0f32; for j in 0..32 { m = m.max(w(r, b * 32 + j).abs()); }
        let mut unit = 2f32.powi(scales[si] as i32 - 127);
        if m / unit > 464.0 { let need = (m / 448.0 / unit).log2().ceil() as i32; scales[si] = (scales[si] as i32 + need).min(254) as u8;
            unit = 2f32.powi(scales[si] as i32 - 127); st.blocks_rescaled += 1; }
        for j in 0..32 { let c = b * 32 + j; let x = w(r, c); let i = r * cols + c;
            let old = vals[i]; st.max_abs_dev = st.max_abs_dev.max((e4m3_dec(old) * unit - x).abs());
            let q = e4.enc(x / unit); if q != old { vals[i] = q; st.codes_changed += 1; } }
    } }
    if st.codes_changed > 0 || st.blocks_rescaled > 0 { st.experts_changed = 1; }
    st
}

fn threads() -> usize { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8) }

// one (layer, matrix) group -> output bytes for its tensors
fn expert_group(e4: &E4, base: &mut St, src: &mut St, p: &str, deint_hint: &mut Option<bool>) -> (Vec<(String, Vec<u8>)>, Stat, bool) {
    let wn = format!("{}_weight", p);
    let tw = base.get(&wn).unwrap().clone();
    let ts = base.get(&format!("{}_scale", wn)).unwrap().clone();
    let nv = base.get(&format!("{}_amax", wn)).is_some();
    let srct = src.get(&wn).unwrap_or_else(|| panic!("fine-tune lacks {}", wn)).clone();
    let (e, rows) = (tw.shape[0], tw.shape[1]);
    let cols = if nv { tw.shape[2] * 2 } else { tw.shape[2] };
    assert_eq!(srct.shape, vec![e, rows, cols], "{}: fine-tune shape {:?} vs base {:?}", wn, srct.shape, tw.shape);
    assert_eq!(srct.dtype, "BF16", "{}: fine-tune dtype {}", wn, srct.dtype);
    let mut w = base.bytes(&wn); let mut s = base.bytes(&ts.name);
    let mut amax: Vec<f32> = if nv { base.bytes(&format!("{}_amax", wn)).chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() } else { vec![] };
    let den = if nv { let b = base.bytes(&format!("{}_outer_scale_denominator", wn)); f32::from_le_bytes([b[0], b[1], b[2], b[3]]) } else { 0.0 };
    let sb = src.bytes(&wn);
    let (wc, sc, srcx) = (w.len() / e, s.len() / e, sb.len() / e);
    // row layout: decided once on the first group (expert 0 reproduced better as stored or de-interleaved)
    let deint = match *deint_hint { Some(d) => d, None => {
        let try_ = |d: bool| { let (mut w0, mut s0, mut a0) = (w[..wc].to_vec(), s[..sc].to_vec(), if nv { amax[0] } else { 0.0 });
            let st = if nv { nv_expert(e4, &mut w0, &mut s0, &mut a0, den, &sb[..srcx], rows, cols, d) } else { mx_expert(e4, &mut w0, &mut s0, &sb[..srcx], rows, cols, d) };
            st.codes_changed };
        let (a, b) = (try_(false), if p.ends_with("w13") { try_(true) } else { u64::MAX });
        println!("  layout probe on {} expert 0: as stored {} codes changed, de-interleaved {}", p, a, if b == u64::MAX { "n/a".into() } else { b.to_string() });
        let d = b < a; *deint_hint = Some(d); d } };
    let t = threads(); let per = (e + t - 1) / t;
    let mut total = Stat::default();
    std::thread::scope(|sc_| {
        let mut hs = vec![];
        let mut amax_chunks: Vec<&mut [f32]> = if nv { amax.chunks_mut(per).collect() } else { vec![] };
        for (k, ((wch, sch), srch)) in w.chunks_mut(per * wc).zip(s.chunks_mut(per * sc)).zip(sb.chunks(per * srcx)).enumerate() {
            let ach = if nv { Some(std::mem::take(&mut amax_chunks[k])) } else { None };
            let e4 = &e4;
            hs.push(sc_.spawn(move || { let mut st = Stat::default(); let n = wch.len() / wc;
                let mut ach = ach;
                for x in 0..n {
                    let (wx, sx, srx) = (&mut wch[x * wc..(x + 1) * wc], &mut sch[x * sc..(x + 1) * sc], &srch[x * srcx..(x + 1) * srcx]);
                    let one = if let Some(a) = ach.as_mut() { nv_expert(e4, wx, sx, &mut a[x], den, srx, rows, cols, deint) } else { mx_expert(e4, wx, sx, srx, rows, cols, deint) };
                    st.add(one);
                } st }));
        }
        for h in hs { total.add(h.join().unwrap()); }
    });
    let mut out = vec![(wn.clone(), w), (ts.name.clone(), s)];
    if nv { out.push((format!("{}_amax", wn), amax.iter().flat_map(|a| a.to_le_bytes()).collect()));
            out.push((format!("{}_outer_scale_denominator", wn), base.bytes(&format!("{}_outer_scale_denominator", wn)))); }
    (out, total, nv)
}

fn convert_plain(bt: &T, src: &mut St) -> (Vec<u8>, &'static str) {
    let Some(stt) = src.get(&bt.name).cloned() else { return (vec![], "missing") };
    let n: usize = bt.shape.iter().product(); let m: usize = stt.shape.iter().product();
    assert_eq!(n, m, "{}: element count {} vs {}", bt.name, n, m);
    let b = src.bytes(&bt.name);
    match (stt.dtype.as_str(), bt.dtype.as_str()) {
        (x, y) if x == y => (b, "copied"),
        ("BF16", "F32") => ((0..n).flat_map(|i| bf(&b, i).to_le_bytes()).collect(), "bf16->f32"),
        ("F32", "BF16") => (b.chunks(4).flat_map(|c| to_bf16(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect(), "f32->bf16"),
        (x, y) => panic!("{}: no conversion {} -> {}", bt.name, x, y),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 { eprintln!("usage: inkconv <base dir> <fine-tune dir> <out dir> [check]"); std::process::exit(2); }
    let check = a.get(4).map(|s| s == "check").unwrap_or(false);
    let t0 = Instant::now();
    let mut base = St::open(&format!("{}/model-00001-of-00001.safetensors", a[1]));
    let mut src = St::open(&format!("{}/model-00001-of-00001.safetensors", a[2]));
    println!("base {} tensors, fine-tune {} tensors, {} threads", base.ts.len(), src.ts.len(), threads());
    let e4 = E4::new();
    let groups: Vec<String> = base.ts.iter().filter(|t| t.name.ends_with(".mlp.experts.w13_weight") || t.name.ends_with(".mlp.experts.w2_weight"))
        .map(|t| t.name.trim_end_matches("_weight").to_string()).collect();
    let is_group_part = |n: &str| groups.iter().any(|g| n.starts_with(g.as_str()) && n[g.len()..].starts_with("_weight"));
    let src_names: std::collections::HashSet<String> = src.ts.iter().map(|t| t.name.clone()).collect();
    let base_names: std::collections::HashSet<String> = base.ts.iter().map(|t| t.name.clone()).collect();
    let extra: Vec<&String> = src_names.iter().filter(|n| !base_names.contains(*n)).collect();
    if !extra.is_empty() { println!("fine-tune tensors not in the base ({}): {:?}", extra.len(), extra.iter().take(8).collect::<Vec<_>>()); }
    // output header in base order with fresh offsets (sizes match the base: same dtypes and shapes)
    let mut hdr = vec![];
    if let Some(m) = &base.meta { hdr.push(("__metadata__".to_string(), m.clone())); }
    let mut off = 0u64;
    for t in &base.ts { let n = t.b - t.a;
        hdr.push((t.name.clone(), J::Obj(vec![("dtype".into(), J::Str(t.dtype.clone())), ("shape".into(), J::Arr(t.shape.iter().map(|x| J::Num(x.to_string())).collect())),
            ("data_offsets".into(), J::Arr(vec![J::Num(off.to_string()), J::Num((off + n).to_string())]))]))); off += n; }
    let mut hs = dump(&J::Obj(hdr)).into_bytes(); while hs.len() % 8 != 0 { hs.push(b' '); }
    let mut out = if check { None } else {
        std::fs::create_dir_all(&a[3]).unwrap();
        let mut f = BufWriter::with_capacity(64 << 20, File::create(format!("{}/model-00001-of-00001.safetensors", a[3])).unwrap());
        f.write_all(&(hs.len() as u64).to_le_bytes()).unwrap(); f.write_all(&hs).unwrap(); Some(f) };
    let mut stash: HashMap<String, Vec<u8>> = HashMap::new();
    let (mut deint, mut counts): (Option<bool>, HashMap<&'static str, usize>) = (None, HashMap::new());
    let (mut nv_tot, mut mx_tot) = (Stat::default(), Stat::default());
    let names: Vec<T> = base.ts.clone();
    for t in &names {
        let bytes = if is_group_part(&t.name) {
            if !stash.contains_key(&t.name) {
                let g = groups.iter().find(|g| t.name.starts_with(g.as_str()) && t.name[g.len()..].starts_with("_weight")).unwrap().clone();
                let tg = Instant::now();
                let (parts, st, nv) = expert_group(&e4, &mut base, &mut src, &g, &mut deint);
                println!("{} [{}] experts changed {}, codes changed {}, blocks rescaled {}, refactored {}, max |fine-tune - base| {:.4e} ({:.1} s)",
                    g, if nv { "nvfp4" } else { "mxfp8" }, st.experts_changed, st.codes_changed, st.blocks_rescaled, st.experts_refactored, st.max_abs_dev, tg.elapsed().as_secs_f32());
                if nv { nv_tot.add(st) } else { mx_tot.add(st) }
                for (n, b) in parts { stash.insert(n, b); }
            }
            stash.remove(&t.name).unwrap()
        } else {
            let (b, how) = convert_plain(t, &mut src);
            *counts.entry(how).or_insert(0) += 1;
            if how == "missing" { println!("  {} not in the fine-tune: base bytes kept", t.name); base.bytes(&t.name) } else { b }
        };
        assert_eq!(bytes.len() as u64, t.b - t.a, "{}: size", t.name);
        if let Some(f) = out.as_mut() { f.write_all(&bytes).unwrap(); }
    }
    if let Some(mut f) = out { f.flush().unwrap(); }
    println!("NVFP4 total: experts changed {}, codes changed {}, blocks rescaled {}, refactored {}", nv_tot.experts_changed, nv_tot.codes_changed, nv_tot.blocks_rescaled, nv_tot.experts_refactored);
    println!("MXFP8 total: experts changed {}, codes changed {}, blocks rescaled {}", mx_tot.experts_changed, mx_tot.codes_changed, mx_tot.blocks_rescaled);
    println!("other tensors: {:?}; w13 rows {}", counts, if deint == Some(true) { "de-interleaved in the fine-tune" } else { "as stored" });
    println!("{} in {:.0} s", if check { "checked" } else { "written" }, t0.elapsed().as_secs_f32());
}
