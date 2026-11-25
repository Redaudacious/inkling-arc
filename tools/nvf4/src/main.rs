// nvf4: host tool for the NVFP4 / MXFP8 DPAS expert kernels on the Arc Pro B70 (Fable, 22 Nov 2026).
// Rust, no dependencies. The GPU kernels are SYCL/ESIMD C++ in libnvf4c.so (C ABI); this tool only moves bytes,
// drives the kernels, checks them against a float reference and times them.
//   nvf4 convert <checkpoint.safetensors> <pack dir> <layer|a-b> [s8] [s16]
//   nvf4 test    <checkpoint.safetensors> <pack dir> <layer> <results.json> [quick] [nopack] [noonehot] [nogemm]
use std::ffi::{c_char, c_int, c_void, CString};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::time::Instant;

extern "C" {
    fn dlopen(f: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(h: *mut c_void, s: *const c_char) -> *mut c_void;
}

type P = *mut c_void;
struct Lib {
    init: unsafe extern "C" fn(*mut c_char, c_int) -> c_int,
    alloc: unsafe extern "C" fn(usize) -> P,
    free: unsafe extern "C" fn(P),
    h2d: unsafe extern "C" fn(P, *const c_void, usize),
    d2h: unsafe extern "C" fn(*mut c_void, P, usize),
    wait: unsafe extern "C" fn(),
    pack: unsafe extern "C" fn(P, P, P, c_int, c_int, c_int, c_int),
    unpack: unsafe extern "C" fn(P, c_int, c_int, c_int, c_int, P),
    mx_pack: unsafe extern "C" fn(P, P, P, P, c_int, c_int, c_int),
    mx_unpack: unsafe extern "C" fn(P, c_int, c_int, c_int, P),
    gemm: unsafe extern "C" fn(P, P, P, P, P, P, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int),
    mx_gemm: unsafe extern "C" fn(P, P, P, P, P, P, c_int, c_int, c_int, c_int, c_int, c_int, c_int),
    // the vLLM port's NVFP4 grouped GEMM behind a C wrapper (libportgemm.so), optional: NVF4_PORT_LIB
    port: Option<unsafe extern "C" fn(P, P, P, P, P, c_int, c_int, P, c_int, c_int, P)>,
}

fn sym(h: *mut c_void, n: &str) -> *mut c_void {
    let c = CString::new(n).unwrap();
    let p = unsafe { dlsym(h, c.as_ptr()) };
    assert!(!p.is_null(), "symbol {} missing", n);
    p
}

fn load_lib() -> Lib {
    let path = std::env::var("NVF4_LIB").unwrap_or_else(|_| "/opt/nvf4/libnvf4c.so".into());
    let c = CString::new(path.clone()).unwrap();
    let h = unsafe { dlopen(c.as_ptr(), 2) };
    assert!(!h.is_null(), "cannot load {}", path);
    unsafe {
        Lib {
            init: std::mem::transmute(sym(h, "nvf4_init")),
            alloc: std::mem::transmute(sym(h, "nvf4_alloc")),
            free: std::mem::transmute(sym(h, "nvf4_free")),
            h2d: std::mem::transmute(sym(h, "nvf4_h2d")),
            d2h: std::mem::transmute(sym(h, "nvf4_d2h")),
            wait: std::mem::transmute(sym(h, "nvf4_wait")),
            pack: std::mem::transmute(sym(h, "nvf4_pack")),
            unpack: std::mem::transmute(sym(h, "nvf4_unpack")),
            mx_pack: std::mem::transmute(sym(h, "mx8_pack")),
            mx_unpack: std::mem::transmute(sym(h, "mx8_unpack")),
            gemm: std::mem::transmute(sym(h, "nvf4_gemm")),
            mx_gemm: std::mem::transmute(sym(h, "mx8_gemm")),
            port: std::env::var("NVF4_PORT_LIB").ok().map(|p| {
                let c = CString::new(p.clone()).unwrap();
                let hp = dlopen(c.as_ptr(), 2);
                assert!(!hp.is_null(), "cannot load {}", p);
                std::mem::transmute(sym(hp, "port_gemm_nvfp4"))
            }),
        }
    }
}

impl Lib {
    fn up<T: Copy>(&self, v: &[T]) -> P {
        let n = std::mem::size_of_val(v);
        unsafe {
            let p = (self.alloc)(n.max(64));
            assert!(!p.is_null(), "device allocation of {} bytes failed", n);
            (self.h2d)(p, v.as_ptr() as *const c_void, n);
            p
        }
    }
    fn dev(&self, bytes: usize) -> P {
        let p = unsafe { (self.alloc)(bytes.max(64)) };
        assert!(!p.is_null(), "device allocation of {} bytes failed", bytes);
        p
    }
    fn down<T: Copy + Default>(&self, p: P, n: usize) -> Vec<T> {
        let mut v = vec![T::default(); n];
        unsafe { (self.d2h)(v.as_mut_ptr() as *mut c_void, p, n * std::mem::size_of::<T>()) };
        v
    }
    fn free(&self, p: P) { unsafe { (self.free)(p) } }
}

// ---------------------------------------------------------------- safetensors (header scan, no JSON crate)
struct St { f: File, hdr: String, base: u64 }
struct TInfo { dtype: String, shape: Vec<usize>, a: u64, b: u64 }

impl St {
    fn open(path: &str) -> St {
        let mut f = File::open(path).expect("open checkpoint");
        let mut l = [0u8; 8];
        f.read_exact(&mut l).unwrap();
        let n = u64::from_le_bytes(l);
        let mut h = vec![0u8; n as usize];
        f.read_exact(&mut h).unwrap();
        St { f, hdr: String::from_utf8(h).unwrap(), base: 8 + n }
    }
    fn info(&self, name: &str) -> Option<TInfo> {
        let key = format!("\"{}\":{{", name);
        let i = self.hdr.find(&key)? + key.len();
        let obj = &self.hdr[i..i + self.hdr[i..].find('}')?];
        let field = |k: &str| -> &str {
            let j = obj.find(&format!("\"{}\":", k)).unwrap() + k.len() + 3;
            let rest = &obj[j..];
            if rest.starts_with('[') { &rest[1..rest.find(']').unwrap()] } else { let r = &rest[1..]; &r[..r.find('"').unwrap()] }
        };
        let nums = |s: &str| -> Vec<u64> { s.split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().unwrap()).collect() };
        let off = nums(field("data_offsets"));
        Some(TInfo { dtype: field("dtype").to_string(), shape: nums(field("shape")).iter().map(|&x| x as usize).collect(), a: off[0], b: off[1] })
    }
    fn bytes(&mut self, t: &TInfo) -> Vec<u8> {
        let mut v = vec![0u8; (t.b - t.a) as usize];
        self.f.seek(SeekFrom::Start(self.base + t.a)).unwrap();
        self.f.read_exact(&mut v).unwrap();
        v
    }
    fn f32s(&mut self, name: &str) -> Vec<f32> {
        let t = self.info(name).unwrap();
        self.bytes(&t).chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
}

// (E, 2I, C) rows with gate/up interleaved -> [gate; up], as the vLLM port stores w13
fn deinterleave(src: &[u8], e: usize, rows: usize, c: usize) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    let half = rows / 2;
    for x in 0..e {
        for r in 0..rows {
            let nr = if r % 2 == 0 { r / 2 } else { half + r / 2 };
            out[(x * rows + nr) * c..(x * rows + nr + 1) * c].copy_from_slice(&src[(x * rows + r) * c..(x * rows + r + 1) * c]);
        }
    }
    out
}

struct Layer { nv: bool, e: usize, n: usize, k: usize, w: Vec<u8>, s: Vec<u8>, fac: Vec<f32>, smin: Vec<i32> }

fn load_layer(st: &mut St, l: usize, name: &str) -> Layer {
    let p = format!("model.llm.layers.{}.mlp.experts.{}_weight", l, name);
    let tw = st.info(&p).expect("weight tensor");
    let ts = st.info(&format!("{}_scale", p)).unwrap();
    let nv = st.info(&format!("{}_amax", p)).is_some();
    let (e, n, wc, sc) = (tw.shape[0], tw.shape[1], tw.shape[2], ts.shape[2]);
    let (mut w, mut s) = (st.bytes(&tw), st.bytes(&ts));
    if name == "w13" { w = deinterleave(&w, e, n, wc); s = deinterleave(&s, e, n, sc); }
    if nv {
        let amax = st.f32s(&format!("{}_amax", p));
        let den = st.f32s(&format!("{}_outer_scale_denominator", p))[0];
        Layer { nv, e, n, k: wc * 2, w, s, fac: amax.iter().map(|a| a / den).collect(), smin: vec![] }
    } else {
        assert!(tw.dtype == "F8_E4M3");
        let per = n * sc;
        let smin: Vec<i32> = (0..e).map(|x| *s[x * per..(x + 1) * per].iter().min().unwrap() as i32).collect();
        let span = (0..e).map(|x| *s[x * per..(x + 1) * per].iter().max().unwrap() as i32 - smin[x]).max().unwrap();
        assert!(span <= 15, "MXFP8 scale span {} > 15", span);
        let fac = smin.iter().map(|&m| 2f32.powi(m - 127)).collect();
        Layer { nv, e, n, k: wc, w, s, fac, smin }
    }
}

const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
fn e4m3(v: u8) -> f32 {
    let (s, e, m) = (v >> 7, ((v >> 3) & 15) as i32, (v & 7) as f32);
    let x = if e == 0 { m / 8.0 * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) };
    if s != 0 { -x } else { x }
}

// reference weights of one expert straight from the checkpoint bytes, without the expert factor (exact in f32)
fn ref_expert(ly: &Layer, x: usize) -> Vec<f32> {
    let (n, k) = (ly.n, ly.k);
    let mut out = vec![0f32; n * k];
    if ly.nv {
        let (wc, sc) = (k / 2, k / 16);
        for r in 0..n {
            let wb = &ly.w[(x * n + r) * wc..(x * n + r + 1) * wc];
            let sb = &ly.s[(x * n + r) * sc..(x * n + r + 1) * sc];
            for c in 0..k {
                let nib = if c & 1 == 1 { wb[c / 2] >> 4 } else { wb[c / 2] & 15 };
                out[r * k + c] = E2M1[nib as usize] * e4m3(sb[c / 16]);
            }
        }
    } else {
        let sc = k / 32;
        for r in 0..n {
            let wb = &ly.w[(x * n + r) * k..(x * n + r + 1) * k];
            let sb = &ly.s[(x * n + r) * sc..(x * n + r + 1) * sc];
            for c in 0..k { out[r * k + c] = e4m3(wb[c]) * 2f32.powi(sb[c / 32] as i32 - ly.smin[x]); }
        }
    }
    out
}

// ---------------------------------------------------------------- pack files
const HDR: usize = 4096;
fn pack_path(dir: &str, l: usize, name: &str, md: &str) -> String { format!("{}/L{}.{}.{}.nvf4", dir, l, name, md) }

fn write_pack(path: &str, meta: &str, pack: &[u8], fac: &[f32]) {
    let mut f = File::create(path).unwrap();
    let mut h = format!("NVF4DPAS\n{}", meta).into_bytes();
    assert!(h.len() <= HDR);
    h.resize(HDR, b' ');
    f.write_all(&h).unwrap();
    f.write_all(pack).unwrap();
    for x in fac { f.write_all(&x.to_le_bytes()).unwrap(); }
}

fn read_pack(path: &str) -> Option<(Vec<u8>, Vec<f32>, usize)> {
    let mut f = File::open(path).ok()?;
    let mut h = vec![0u8; HDR];
    f.read_exact(&mut h).unwrap();
    let hs = String::from_utf8_lossy(&h).to_string();
    assert!(hs.starts_with("NVF4DPAS"));
    let get = |k: &str| -> usize { hs.lines().find(|l| l.starts_with(&format!("{}=", k))).unwrap()[k.len() + 1..].trim().parse().unwrap() };
    let (nb, e, rec) = (get("pack_bytes"), get("E"), get("rec"));
    let mut pack = vec![0u8; nb];
    f.read_exact(&mut pack).unwrap();
    let mut fb = vec![0u8; e * 4];
    f.read_exact(&mut fb).unwrap();
    Some((pack, fb.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(), rec))
}

fn convert(lib: &Lib, ckpt: &str, dir: &str, spec: &str, modes: &[String]) {
    let mut st = St::open(ckpt);
    std::fs::create_dir_all(dir).unwrap();
    let (a, b) = match spec.split_once('-') { Some((a, b)) => (a.parse().unwrap(), b.parse().unwrap()), None => (spec.parse().unwrap(), spec.parse::<usize>().unwrap()) };
    let t00 = Instant::now();
    for l in a..=b {
        for name in ["w13", "w2"] {
            let t0 = Instant::now();
            let ly = load_layer(&mut st, l, name);
            let (e, n, k) = (ly.e, ly.n, ly.k);
            let (dw, ds) = (lib.up(&ly.w), lib.up(&ly.s));
            let mds: Vec<&str> = if ly.nv { modes.iter().map(|s| s.as_str()).collect() } else { vec!["mx8"] };
            for md in mds {
                let rec = match md { "s8" => 576, "s16" => 640, _ => 1056 };
                let nb = e * (n / 16) * (k / 64) * rec;
                let out = lib.dev(nb);
                unsafe {
                    if ly.nv { (lib.pack)(dw, ds, out, e as c_int, n as c_int, k as c_int, (md == "s8") as c_int); }
                    else { let sm = lib.up(&ly.smin); (lib.mx_pack)(dw, ds, sm, out, e as c_int, n as c_int, k as c_int); lib.free(sm); }
                }
                let pack: Vec<u8> = lib.down(out, nb);
                lib.free(out);
                let meta = format!("layer={}\nname={}\nformat={}\nscale={}\nE={}\nN={}\nK={}\nrec={}\npack_bytes={}\nrows={}\n", l, name,
                    if ly.nv { "nvfp4" } else { "mxfp8" }, md, e, n, k, rec, nb, if name == "w13" { "gate_then_up" } else { "as_shipped" });
                write_pack(&pack_path(dir, l, name, md), &meta, &pack, &ly.fac);
                println!("layer {} {} {}: [{},{},{}] shipped {:.1} MB -> pack {:.1} MB ({:.3} bit/weight) {:.1}s", l, name, md, e, n, k,
                    (ly.w.len() + ly.s.len()) as f64 / 1e6, nb as f64 / 1e6, nb as f64 * 8.0 / (e * n * k) as f64, t0.elapsed().as_secs_f64());
            }
            lib.free(dw); lib.free(ds);
        }
    }
    println!("total {:.1}s", t00.elapsed().as_secs_f64());
}

// ---------------------------------------------------------------- small numeric helpers
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
    fn unit(&mut self) -> f64 { ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64 }
    fn normal(&mut self) -> f32 { ((-2.0 * self.unit().ln()).sqrt() * (2.0 * std::f64::consts::PI * self.unit()).cos()) as f32 }
}
fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 31) as i32, (h & 1023) as f32);
    let v = if e == 0 { m * 2f32.powi(-24) } else if e == 31 { f32::INFINITY } else { (1.0 + m / 1024.0) * 2f32.powi(e - 15) };
    if s != 0 { -v } else { v }
}
// value with 8 significant bits (bf16-like) and |x| in [2^-8, 2^8): exact in bf16 and in fp16
fn act(r: &mut Rng) -> (f32, u16, u16) {
    let x = r.normal().clamp(-200.0, 200.0);
    if x.abs() < 0.00390625 { return (0.0, 0, 0); }
    let b = (x.to_bits() + 0x8000) & 0xFFFF_0000;
    let v = f32::from_bits(b);
    let (e, m) = (((b >> 23) & 255) as i32 - 127, (b >> 13) & 1023);
    (v, (((b >> 31) as u16) << 15) | (((e + 15) as u16) << 10) | m as u16, (b >> 16) as u16)
}

struct Routing { jobs: Vec<i32>, nj: usize, rows: usize, distinct: usize, cnt: Vec<i32> }
fn routing(t: usize, seed: u64) -> Routing {
    let mut r = Rng(seed);
    let mut cnt = [0usize; 128];
    for _ in 0..t {
        let mut ids: Vec<usize> = (0..128).collect();
        for i in 0..8 { let j = i + r.below(128 - i); ids.swap(i, j); cnt[ids[i]] += 1; }
    }
    let (mut jobs, mut row) = (vec![], 0usize);
    for (e, &c) in cnt.iter().enumerate() {
        let mut d = 0;
        while d < c { let m = (c - d).min(8); jobs.extend_from_slice(&[e as i32, (row + d) as i32, m as i32, 0]); d += m; }
        row += c;
    }
    Routing { nj: jobs.len() / 4, jobs, rows: t * 8, distinct: cnt.iter().filter(|&&c| c > 0).count(), cnt: cnt.iter().map(|&c| c as i32).collect() }
}

fn test(lib: &Lib, ckpt: &str, dir: &str, l: usize, out_json: &str, flags: &[String]) {
    let has = |f: &str| flags.iter().any(|x| x == f);
    let quick = has("quick");
    let t00 = Instant::now();
    let mut st = St::open(ckpt);
    let mut js = format!("{{\"layer\":{},\"tool\":\"rust\"", l);
    let (mut jpack, mut jone, mut jgemm) = (vec![], vec![], vec![]);
    for name in ["w13", "w2"] {
        let ly = load_layer(&mut st, l, name);
        let (n, k) = (ly.n, ly.k);
        let mds: Vec<&str> = if ly.nv { vec!["s8", "s16"] } else { vec!["mx8"] };
        let mut packs = vec![];
        for md in mds {
            if let Some((pk, fac, rec)) = read_pack(&pack_path(dir, l, name, md)) {
                assert!(fac == ly.fac, "factors in the pack differ from the checkpoint");
                packs.push((md, lib.up(&pk), rec));
            }
        }
        let dfac = lib.up(&ly.fac);
        let (dw, ds) = (lib.up(&ly.w), lib.up(&ly.s));   // the shipped bytes, for the port's kernel
        let ones = lib.up(&vec![1f32; ly.e]);
        println!("[{:6.1}s] {} layer {} {}: [{},{},{}] packs {:?}", t00.elapsed().as_secs_f64(), if ly.nv { "NVFP4" } else { "MXFP8" }, l, name,
            ly.e, n, k, packs.iter().map(|p| p.0).collect::<Vec<_>>());
        // 1) pack check: scalar unpack of the on-disk pack == checkpoint values, bit for bit
        if !has("nopack") {
            let dout = lib.dev(n * k * 4);
            for &(md, pk, _) in &packs {
                let (mut bad, mut tot) = (0usize, 0usize);
                let list: Vec<usize> = if quick { vec![0, 57, 127] } else { (0..128).step_by(9).collect() };
                for &x in &list {
                    unsafe { if ly.nv { (lib.unpack)(pk, x as c_int, n as c_int, k as c_int, (md == "s8") as c_int, dout) } else { (lib.mx_unpack)(pk, x as c_int, n as c_int, k as c_int, dout) } }
                    let got: Vec<f32> = lib.down(dout, n * k);
                    let rf = ref_expert(&ly, x);
                    bad += got.iter().zip(&rf).filter(|(a, b)| a != b).count();
                    tot += rf.len();
                }
                println!("[{:6.1}s] pack check {} {}: {} of {} weights differ", t00.elapsed().as_secs_f64(), name, md, bad, tot);
                jpack.push(format!("{{\"name\":\"{}\",\"scale\":\"{}\",\"differ\":{},\"checked\":{}}}", name, md, bad, tot));
            }
            lib.free(dout);
        }
        let c16 = lib.dev(256 * n * 2);
        let c32 = lib.dev(256 * n * 4);
        let call = |pk: P, md: &str, a: P, f: P, jobs: P, rows: usize, nj: usize, split: usize, mode: usize, big: usize| unsafe {
            if ly.nv { (lib.gemm)(a, pk, f, jobs, c16, c32, rows as c_int, nj as c_int, n as c_int, k as c_int, split as c_int, (md == "s8") as c_int, mode as c_int, big as c_int) }
            else { (lib.mx_gemm)(a, pk, f, jobs, c16, c32, rows as c_int, nj as c_int, n as c_int, k as c_int, split as c_int, mode as c_int, big as c_int) }
        };
        let result = |rows: usize, split: usize| -> Vec<f32> {
            if split > 1 { lib.down::<f32>(c32, rows * n) } else { lib.down::<u16>(c16, rows * n).iter().map(|&h| f16_to_f32(h)).collect() }
        };
        // 2) one-hot activations through the DPAS kernel: the weights must come out bit for bit (expert factor 1)
        if !has("noonehot") {
            for &(md, pk, _) in &packs {
                for mode in 0..2 {
                    let (mut bad, mut tot, mut mx) = (0usize, 0usize, 0f64);
                    let mut r = Rng(77 + mode as u64);
                    for &x in &[0usize, 57, 127] {
                        let rf = ref_expert(&ly, x);
                        for _ in 0..(if quick { 2 } else { 8 }) {
                            let ks: Vec<usize> = (0..8).map(|_| r.below(k)).collect();
                            let mut a = vec![0u16; 8 * k];
                            for i in 0..8 { a[i * k + ks[i]] = 0x3C00; }
                            let (da, dj) = (lib.up(&a), lib.up(&[x as i32, 0, 8, 0]));
                            // NVFP4: fp16 store (split 1). MXFP8 values can leave the fp16 range, so read the fp32 sums (split 2).
                            let split = if ly.nv { 1 } else { 2 };
                            call(pk, md, da, ones, dj, 8, 1, split, mode, 0);
                            let got = result(8, split);
                            for i in 0..8 { for c in 0..n {
                                let (g, w) = (got[i * n + c], rf[c * k + ks[i]]);
                                if g != w { bad += 1; mx = mx.max(((g - w).abs() / w.abs().max(1e-30)) as f64); }
                                tot += 1;
                            } }
                            lib.free(da); lib.free(dj);
                        }
                    }
                    println!("[{:6.1}s] one-hot {} {} mode{}: {} of {} differ, max rel diff {:.3e}", t00.elapsed().as_secs_f64(), name, md, mode, bad, tot, mx);
                    jone.push(format!("{{\"name\":\"{}\",\"scale\":\"{}\",\"mode\":{},\"differ\":{},\"checked\":{},\"maxrel\":{:e}}}", name, md, mode, bad, tot, mx));
                }
            }
        }
        // 3) grouped GEMM with top-8 of 128 routing: rel.err vs f64 reference, time, effective GB/s over the shipped bytes
        if !has("nogemm") {
            let splits: Vec<usize> = [1usize, 2, 3, 4, 6, 9, 12].iter().cloned().filter(|s| (k / 64) % s == 0).collect();
            for &t in &[1usize, 4, 8, 16, 32] {
                let rt = routing(t, 1234 + t as u64);
                let mut r = Rng(99 + t as u64);
                let (mut av, mut ah, mut ab) = (vec![0f32; rt.rows * k], vec![0u16; rt.rows * k], vec![0u16; rt.rows * k]);
                for i in 0..rt.rows * k { let (v, h, b) = act(&mut r); av[i] = v; ah[i] = h; ab[i] = b; }
                let mut rf = vec![0f64; rt.rows * n];
                for j in 0..rt.nj {
                    let (x, r0, m) = (rt.jobs[j * 4] as usize, rt.jobs[j * 4 + 1] as usize, rt.jobs[j * 4 + 2] as usize);
                    let w = ref_expert(&ly, x);
                    for i in r0..r0 + m { for c in 0..n {
                        let mut s = 0f64;
                        for q in 0..k { s += av[i * k + q] as f64 * w[c * k + q] as f64; }
                        rf[i * n + c] = s * ly.fac[x] as f64;
                    } }
                }
                let rn = rf.iter().map(|x| x * x).sum::<f64>().sqrt();
                let (da, dj) = (lib.up(&ah), lib.up(&rt.jobs));
                // timing rotates over 16 different routings (same row count) so no call re-reads the experts of the call before
                let vars: Vec<Routing> = (0..16).map(|v| routing(t, 5000 + 100 * t as u64 + v)).collect();
                let vj: Vec<(P, usize)> = vars.iter().map(|v| (lib.up(&v.jobs), v.nj)).collect();
                let vc: Vec<P> = vars.iter().map(|v| lib.up(&v.cnt)).collect();
                let avg_distinct = vars.iter().map(|v| v.distinct).sum::<usize>() as f64 / 16.0;
                let native = (avg_distinct * if ly.nv { (n * k / 2 + n * k / 16) as f64 } else { (n * k + n * k / 32) as f64 }) as usize;
                for &(md, pk, rec) in &packs {
                    for mode in 0..2 {
                        let mut best: Option<(f64, usize, usize, f64)> = None;
                        for &sp in &splits { for big in 0..2 {
                            call(pk, md, da, dfac, dj, rt.rows, rt.nj, sp, mode, big);
                            let got = result(rt.rows, sp);
                            let rel = got.iter().zip(&rf).map(|(a, b)| (*a as f64 - b).powi(2)).sum::<f64>().sqrt() / rn;
                            for i in 0..16 { call(pk, md, da, dfac, vj[i].0, rt.rows, vj[i].1, sp, mode, big); }
                            unsafe { (lib.wait)() };
                            let it = if quick { 16 } else { 48 };
                            let t0 = Instant::now();
                            for i in 0..it { call(pk, md, da, dfac, vj[i % 16].0, rt.rows, vj[i % 16].1, sp, mode, big); }
                            unsafe { (lib.wait)() };
                            let dt = t0.elapsed().as_secs_f64() / it as f64;
                            if best.map_or(true, |b| dt < b.0) { best = Some((dt, sp, big, rel)); }
                        } }
                        let (dt, sp, big, rel) = best.unwrap();
                        let read = (avg_distinct * (rec * (n / 16) * (k / 64)) as f64) as usize;
                        println!("[{:6.1}s] T={:2} rows={:3} experts={:5.1} {} {} mode{}: rel.err {:.2e}  {:8.1} us  shipped-bytes {:6.1} GB/s  read {:6.1} GB/s  (split {}, grf256 {})",
                            t00.elapsed().as_secs_f64(), t, rt.rows, avg_distinct, name, md, mode, rel, dt * 1e6, native as f64 / dt / 1e9, read as f64 / dt / 1e9, sp, big);
                        jgemm.push(format!("{{\"T\":{},\"rows\":{},\"experts\":{:.1},\"name\":\"{}\",\"scale\":\"{}\",\"mode\":{},\"rel\":{:e},\"us\":{:.2},\"native_gbs\":{:.2},\"read_gbs\":{:.2},\"split\":{},\"big\":{},\"native_bytes\":{}}}",
                            t, rt.rows, avg_distinct, name, md, mode, rel, dt * 1e6, native as f64 / dt / 1e9, read as f64 / dt / 1e9, sp, big, native));
                    }
                }
                // the port's kernel on the shipped bytes: same activations (as bf16), same routing, same timing loop
                if let (Some(pf), true) = (lib.port, ly.nv) {
                    let (dab, dcnt, dd, dat) = (lib.up(&ab), lib.up(&rt.cnt), lib.dev(rt.rows * n * 2), lib.up(&[0i32; 16]));
                    let run = || unsafe { pf(dab, dw, ds, dfac, dd, n as c_int, k as c_int, dcnt, ly.e as c_int, rt.rows as c_int, dat) };
                    run();
                    let got: Vec<u16> = lib.down(dd, rt.rows * n);
                    let rel = got.iter().zip(&rf).map(|(a, b)| (f32::from_bits((*a as u32) << 16) as f64 - b).powi(2)).sum::<f64>().sqrt() / rn;
                    let runv = |i: usize| unsafe { pf(dab, dw, ds, dfac, dd, n as c_int, k as c_int, vc[i % 16], ly.e as c_int, rt.rows as c_int, dat) };
                    for i in 0..16 { runv(i); }
                    unsafe { (lib.wait)() };
                    let it = if quick { 16 } else { 48 };
                    let t0 = Instant::now();
                    for i in 0..it { runv(i); }
                    unsafe { (lib.wait)() };
                    let dt = t0.elapsed().as_secs_f64() / it as f64;
                    println!("[{:6.1}s] T={:2} rows={:3} experts={:5.1} {} PORT kernel: rel.err {:.2e}  {:8.1} us  shipped-bytes {:6.1} GB/s",
                        t00.elapsed().as_secs_f64(), t, rt.rows, avg_distinct, name, rel, dt * 1e6, native as f64 / dt / 1e9);
                    jgemm.push(format!("{{\"T\":{},\"rows\":{},\"experts\":{:.1},\"name\":\"{}\",\"scale\":\"port\",\"mode\":-1,\"rel\":{:e},\"us\":{:.2},\"native_gbs\":{:.2},\"read_gbs\":{:.2},\"split\":0,\"big\":1,\"native_bytes\":{}}}",
                        t, rt.rows, avg_distinct, name, rel, dt * 1e6, native as f64 / dt / 1e9, native as f64 / dt / 1e9, native));
                    lib.free(dab); lib.free(dcnt); lib.free(dd); lib.free(dat);
                }
                lib.free(da); lib.free(dj);
                for &(p, _) in &vj { lib.free(p); }
                for &p in &vc { lib.free(p); }
            }
        }
        lib.free(dw); lib.free(ds);
        for &(_, pk, _) in &packs { lib.free(pk); }
        lib.free(dfac); lib.free(ones); lib.free(c16); lib.free(c32);
    }
    js += &format!(",\"pack\":[{}],\"onehot\":[{}],\"gemm\":[{}]}}\n", jpack.join(","), jone.join(","), jgemm.join(","));
    File::create(out_json).unwrap().write_all(js.as_bytes()).unwrap();
    println!("DONE {:.1}s", t00.elapsed().as_secs_f64());
}


// ---------------------------------------------------------------- whole expert layer (nvf4_moe) vs f64 reference
type MoeFn = unsafe extern "C" fn(P, P, P, P, P, P, P, P, P, P, P, P, P, P, c_int, c_int, c_int, c_int, c_int, f32, f32, c_int);

fn moe_test(lib: &Lib, ckpt: &str, dir: &str, l: usize, out_json: &str, gc: f32, uc: f32) {
    let path = std::env::var("NVF4_LIB").unwrap_or_else(|_| "/opt/nvf4/libnvf4c.so".into());
    let c = CString::new(path).unwrap();
    let h = unsafe { dlopen(c.as_ptr(), 2) };
    let moe: MoeFn = unsafe { std::mem::transmute(sym(h, "nvf4_moe")) };
    let maxjobs: unsafe extern "C" fn(c_int, c_int) -> c_int = unsafe { std::mem::transmute(sym(h, "nvf4_moe_max_jobs")) };
    let t00 = Instant::now();
    let mut st = St::open(ckpt);
    let (l13, l2) = (load_layer(&mut st, l, "w13"), load_layer(&mut st, l, "w2"));
    let (hd, inter, e, k) = (l13.k, l2.k, l13.e, 8usize);
    let md = if l13.nv { "s8" } else { "mx8" };
    let fmt = if l13.nv { 0 } else { 2 };
    let (p13, f13, _) = read_pack(&pack_path(dir, l, "w13", md)).expect("w13 pack");
    let (p2, f2, _) = read_pack(&pack_path(dir, l, "w2", md)).expect("w2 pack");
    let (dp13, df13, dp2, df2) = (lib.up(&p13), lib.up(&f13), lib.up(&p2), lib.up(&f2));
    drop(p13); drop(p2);
    println!("[{:6.1}s] layer {} {}: H {} I {} E {}", t00.elapsed().as_secs_f64(), l, md, hd, inter, e);
    let mut rows_json = vec![];
    for &t in &[1usize, 4, 8, 32, 64] {
        let rows = t * k;
        let mj = unsafe { maxjobs(rows as c_int, e as c_int) } as usize;
        // 16 routings: variant 0 is checked against the reference, all are used for timing
        let mut r = Rng(4242 + t as u64);
        let mut variants = vec![];
        for _ in 0..16 {
            let (mut ids, mut wt) = (vec![0i32; rows], vec![0f32; rows]);
            for tok in 0..t {
                let mut all: Vec<usize> = (0..e).collect();
                let mut sum = 0f32;
                for s in 0..k { let j = s + r.below(e - s); all.swap(s, j); ids[tok * k + s] = all[s] as i32; wt[tok * k + s] = 0.05 + r.unit() as f32; sum += wt[tok * k + s]; }
                for s in 0..k { wt[tok * k + s] /= sum; }
            }
            variants.push((ids, wt));
        }
        let (mut xv, mut xb) = (vec![0f32; t * hd], vec![0u16; t * hd]);
        for i in 0..t * hd { let (v, _, b) = act(&mut r); xv[i] = v; xb[i] = b; }
        // f64 reference for variant 0, expert by expert
        let (ids0, wt0) = (&variants[0].0, &variants[0].1);
        let mut rf = vec![0f64; t * hd];
        for x in 0..e {
            let pairs: Vec<usize> = (0..rows).filter(|&i| ids0[i] as usize == x).collect();
            if pairs.is_empty() { continue; }
            let (w13, w2) = (ref_expert(&l13, x), ref_expert(&l2, x));
            for &i in &pairs {
                let tok = i / k;
                let mut a = vec![0f64; inter];
                for n in 0..inter {
                    let (mut g, mut u) = (0f64, 0f64);
                    for q in 0..hd { let xq = xv[tok * hd + q] as f64; g += xq * w13[n * hd + q] as f64; u += xq * w13[(inter + n) * hd + q] as f64; }
                    let (g, u) = (g * l13.fac[x] as f64, u * l13.fac[x] as f64);
                    a[n] = gc as f64 * (g / gc as f64).tanh() * (1.0 / (1.0 + (-g).exp())) * uc as f64 * (u / uc as f64).tanh();
                }
                for c in 0..hd {
                    let mut s = 0f64;
                    for n in 0..inter { s += a[n] * w2[c * inter + n] as f64; }
                    rf[tok * hd + c] += wt0[i] as f64 * s * l2.fac[x] as f64;
                }
            }
        }
        let rn = rf.iter().map(|x| x * x).sum::<f64>().sqrt();
        let dx = lib.up(&xb);
        let dv: Vec<(P, P)> = variants.iter().map(|(i, w)| (lib.up(i), lib.up(w))).collect();
        let (unp, jobs, a16, c1) = (lib.dev(rows * 4), lib.dev(mj * 16), lib.dev(rows * hd * 2), lib.dev(rows * 2 * inter * 4));
        let (actb, c2, out) = (lib.dev(rows * inter * 2), lib.dev(rows * hd * 4), lib.dev(t * hd * 2));
        let run = |v: usize| unsafe { moe(dx, dv[v].1, dv[v].0, dp13, df13, dp2, df2, unp, jobs, a16, c1, actb, c2, out, t as c_int, hd as c_int, k as c_int, inter as c_int, e as c_int, gc, uc, fmt) };
        run(0);
        let got: Vec<u16> = lib.down(out, t * hd);
        let rel = got.iter().zip(&rf).map(|(a, b)| (f32::from_bits((*a as u32) << 16) as f64 - b).powi(2)).sum::<f64>().sqrt() / rn;
        for i in 0..16 { run(i); }
        unsafe { (lib.wait)() };
        let it = 48;
        let t0 = Instant::now();
        for i in 0..it { run(i % 16); }
        unsafe { (lib.wait)() };
        let us = t0.elapsed().as_secs_f64() / it as f64 * 1e6;
        println!("[{:6.1}s] T={:2} rows={:3} whole layer: rel.err vs f64 reference {:.2e} (output rounded to bf16)  {:8.1} us", t00.elapsed().as_secs_f64(), t, rows, rel, us);
        rows_json.push(format!("{{\"T\":{},\"rows\":{},\"rel\":{:e},\"us\":{:.2}}}", t, rows, rel, us));
        for p in [dx, unp, jobs, a16, c1, actb, c2, out] { lib.free(p); }
        for (a, b) in dv { lib.free(a); lib.free(b); }
    }
    for p in [dp13, df13, dp2, df2] { lib.free(p); }
    File::create(out_json).unwrap().write_all(format!("{{\"layer\":{},\"format\":\"{}\",\"moe_layer\":[{}]}}\n", l, md, rows_json.join(",")).as_bytes()).unwrap();
    println!("DONE {:.1}s", t00.elapsed().as_secs_f64());
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 { eprintln!("usage: nvf4 convert <ckpt> <pack dir> <layers> [s8] [s16] | nvf4 test <ckpt> <pack dir> <layer> <results.json> [quick]"); std::process::exit(2); }
    let lib = load_lib();
    let mut name = vec![0 as c_char; 256];
    let rc = unsafe { (lib.init)(name.as_mut_ptr(), 256) };
    let dn = unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }.to_string_lossy().to_string();
    eprintln!("device: {}", dn);
    if rc != 0 || !dn.contains("B70") && std::env::var("NVF4_ANY_GPU").is_err() { eprintln!("not the Arc Pro B70; set NVF4_ANY_GPU=1 to override"); std::process::exit(3); }
    match a[1].as_str() {
        "convert" => { let m = if a.len() > 5 { a[5..].to_vec() } else { vec!["s8".to_string()] }; convert(&lib, &a[2], &a[3], &a[4], &m) }
        "test" => test(&lib, &a[2], &a[3], a[4].parse().unwrap(), &a[5], &a[6..]),
        "moe" => moe_test(&lib, &a[2], &a[3], a[4].parse().unwrap(), &a[5], 4.0, 25.0),
        _ => std::process::exit(2),
    }
}
