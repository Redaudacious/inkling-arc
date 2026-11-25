// inkeval: quality evaluation of an OpenAI-compatible chat server (Claude, 23 Nov 2026). Rust, std only.
//   inkeval <host:port> <model> <out.jsonl> [rustc path]
// Cases: instruction following, engineering math, reasoning, tool calls, long-context retrieval (~30K and ~100K
// tokens), Rust coding (the answer's code is compiled with rustc and run against tests), and one picture.
// Every request is non-streamed at temperature 0; every answer is saved to <out.jsonl> so it can be read.
use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::Command;
use std::time::{Duration, Instant};

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

// Parse a JSON string starting at s[i] == '"'; returns (value, index after closing quote).
fn jstr(s: &[u8], mut i: usize) -> Option<(String, usize)> {
    if s.get(i) != Some(&b'"') { return None; }
    i += 1;
    let mut out: Vec<u8> = Vec::new();
    while i < s.len() {
        match s[i] {
            b'"' => return Some((String::from_utf8_lossy(&out).into_owned(), i + 1)),
            b'\\' => {
                i += 1;
                match s.get(i)? {
                    b'n' => out.push(b'\n'), b't' => out.push(b'\t'), b'r' => out.push(b'\r'),
                    b'b' => out.push(8), b'f' => out.push(12), b'/' => out.push(b'/'),
                    b'"' => out.push(b'"'), b'\\' => out.push(b'\\'),
                    b'u' => {
                        let h = std::str::from_utf8(s.get(i + 1..i + 5)?).ok()?;
                        let mut cp = u32::from_str_radix(h, 16).ok()?;
                        i += 4;
                        if (0xD800..0xDC00).contains(&cp) && s.get(i + 1) == Some(&b'\\') && s.get(i + 2) == Some(&b'u') {
                            let h2 = std::str::from_utf8(s.get(i + 3..i + 7)?).ok()?;
                            let lo = u32::from_str_radix(h2, 16).ok()?;
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            i += 6;
                        }
                        let ch = char::from_u32(cp).unwrap_or('?');
                        let mut b = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut b).as_bytes());
                    }
                    c => out.push(*c),
                }
                i += 1;
            }
            c => { out.push(c); i += 1; }
        }
    }
    None
}

// Value of the first `"key":"..."` string after byte offset `from`.
fn field(body: &str, key: &str, from: usize) -> Option<String> {
    let pat = format!("\"{}\":", key);
    let b = body.as_bytes();
    let mut i = body[from..].find(&pat)? + from + pat.len();
    while b.get(i) == Some(&b' ') { i += 1; }
    jstr(b, i).map(|(v, _)| v)
}

fn num(body: &str, key: &str) -> u64 {
    let pat = format!("\"{}\":", key);
    body.rfind(&pat).map(|i| body[i + pat.len()..].chars().skip_while(|c| *c == ' ').take_while(|c| c.is_ascii_digit()).collect::<String>())
        .and_then(|t| t.parse().ok()).unwrap_or(0)
}

fn post(addr: &str, body: &str, timeout: u64) -> Result<String, String> {
    let mut s = TcpStream::connect(addr).map_err(|e| e.to_string())?;
    s.set_read_timeout(Some(Duration::from_secs(timeout))).ok();
    let req = format!("POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}", body.len());
    s.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).map_err(|e| e.to_string())?;
    let txt = String::from_utf8_lossy(&raw).into_owned();
    let split = txt.find("\r\n\r\n").ok_or("no header end")?;
    let (head, rest) = (&txt[..split], &txt[split + 4..]);
    let body = if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        let mut out = String::new();
        let mut r = rest;
        loop {
            let nl = match r.find("\r\n") { Some(n) => n, None => break };
            let n = usize::from_str_radix(r[..nl].trim(), 16).unwrap_or(0);
            if n == 0 { break; }
            let start = nl + 2;
            if start + n > r.len() { out.push_str(&r[start..]); break; }
            out.push_str(&r[start..start + n]);
            r = &r[start + n..];
            if r.starts_with("\r\n") { r = &r[2..]; }
        }
        out
    } else { rest.to_string() };
    if !head.starts_with("HTTP/1.1 200") { return Err(format!("{} {}", head.lines().next().unwrap_or(""), &body[..body.len().min(300)])); }
    Ok(body)
}

// --- tiny PNG (uncompressed deflate) for the picture case ---
fn crc32(d: &[u8]) -> u32 {
    let mut c = 0xffff_ffffu32;
    for &b in d { c ^= b as u32; for _ in 0..8 { c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 }; } }
    !c
}
fn png(w: usize, h: usize, px: &dyn Fn(usize, usize) -> [u8; 3]) -> Vec<u8> {
    let mut raw = Vec::new();
    for y in 0..h { raw.push(0u8); for x in 0..w { raw.extend_from_slice(&px(x, y)); } }
    let mut z = vec![0x78u8, 0x01];
    for (k, ch) in raw.chunks(65535).enumerate() {
        let last = (k + 1) * 65535 >= raw.len();
        z.push(last as u8);
        let n = ch.len() as u16;
        z.extend_from_slice(&n.to_le_bytes());
        z.extend_from_slice(&(!n).to_le_bytes());
        z.extend_from_slice(ch);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in &raw { a = (a + x as u32) % 65521; b = (b + a) % 65521; }
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut chunk = |t: &[u8; 4], d: &[u8]| {
        out.extend_from_slice(&(d.len() as u32).to_be_bytes());
        let mut td = t.to_vec(); td.extend_from_slice(d);
        out.extend_from_slice(&td);
        out.extend_from_slice(&crc32(&td).to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(w as u32).to_be_bytes()); ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(b"IHDR", &ihdr); chunk(b"IDAT", &z); chunk(b"IEND", &[]);
    out
}
fn b64(d: &[u8]) -> String {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::new();
    for c in d.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        s.push(T[(n >> 18) as usize & 63] as char); s.push(T[(n >> 12) as usize & 63] as char);
        s.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        s.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    s
}

fn haystack(entries: usize, at: usize, needle: &str) -> String {
    let mut s = String::from("Below is a construction site log. Read it, then answer the question at the end.\n");
    for i in 1..=entries {
        if i == at { s.push_str(&format!("Log entry {i}: SECURITY NOTE - {needle}\n")); }
        s.push_str(&format!("Log entry {i}: crew checked the scaffold on level {}, weather clear, concrete pour {} of 40 finished, no incidents.\n", i % 12, i % 40));
    }
    s
}

fn rust_block(text: &str) -> Option<String> {
    let start = text.find("```rust").map(|i| i + 7).or_else(|| text.find("```").map(|i| i + 3))?;
    let rest = &text[start..];
    let end = rest.find("```")?;
    Some(rest[..end].to_string())
}

fn run_rust(rustc: &str, name: &str, code: &str, tests: &str) -> (bool, String) {
    let dir = std::env::temp_dir().join("inkeval");
    std::fs::create_dir_all(&dir).ok();
    let src = dir.join(format!("{name}.rs"));
    let exe = dir.join(format!("{name}{}", if cfg!(windows) { ".exe" } else { "" }));
    let full = format!("#![allow(dead_code, unused)]\n{code}\nfn main() {{\n{tests}\nprintln!(\"ok\");\n}}\n");
    if std::fs::write(&src, full).is_err() { return (false, "write failed".into()); }
    let c = Command::new(rustc).arg("-O").arg("--edition").arg("2021").arg("-o").arg(&exe).arg(&src).output();
    match c {
        Ok(o) if o.status.success() => {}
        Ok(o) => return (false, format!("compile: {}", String::from_utf8_lossy(&o.stderr).lines().take(6).collect::<Vec<_>>().join(" | "))),
        Err(e) => return (false, format!("rustc: {e}")),
    }
    match Command::new(&exe).output() {
        Ok(o) if o.status.success() => (true, "tests passed".into()),
        Ok(o) => (false, format!("run: {}", String::from_utf8_lossy(&o.stderr).lines().take(4).collect::<Vec<_>>().join(" | "))),
        Err(e) => (false, format!("run: {e}")),
    }
}

struct Case { id: &'static str, group: &'static str, prompt: String, tools: Option<&'static str>, image: Option<String>, max_tok: u32, check: Check }
enum Check { Exact(&'static str), Contains(Vec<&'static str>), StartsYes, JsonArray3, Bullets3, OneShortSentence, Sorted(&'static str),
             Tool(&'static str, Vec<&'static str>), Rust(&'static str, &'static str) }

const T_WEATHER: &str = r#"[{"type":"function","function":{"name":"get_weather","description":"Current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]"#;
const T_CALC: &str = r#"[{"type":"function","function":{"name":"calculate","description":"Evaluate an arithmetic expression exactly.","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}]"#;
const T_FILE: &str = r#"[{"type":"function","function":{"name":"write_file","description":"Write a text file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]"#;

fn cases() -> Vec<Case> {
    let c = |id, group, p: &str, max_tok, check| Case { id, group, prompt: p.to_string(), tools: None, image: None, max_tok, check };
    let mut v = vec![
        c("I1", "instruction", "Reply with exactly one word: Ready", 600, Check::Exact("ready")),
        c("I2", "instruction", "List three kinds of loads on a building as a JSON array of three lowercase strings. Output only the JSON array.", 800, Check::JsonArray3),
        c("I3", "instruction", "Write exactly three bullet points about curing concrete. Each bullet is one line starting with \"- \". No other text.", 900, Check::Bullets3),
        c("I4", "instruction", "In one sentence of fewer than 20 words, what does a lintel do?", 700, Check::OneShortSentence),
        c("M1", "math", "A simply supported beam 6 m long carries a uniform load of 4 kN/m. What is the maximum bending moment in kN·m? Give the number.", 1500, Check::Contains(vec!["18"])),
        c("M2", "math", "A room is 4.2 m by 3.5 m. What is its floor area in square metres? Give the number.", 900, Check::Contains(vec!["14.7"])),
        c("M3", "math", "A ramp rises 0.5 m over a horizontal run of 6 m. Express the slope as a ratio 1:x.", 1200, Check::Contains(vec!["1:12"])),
        c("M4", "math", "Convert 2,500 square feet to square metres using 1 ft = 0.3048 m. Round to one decimal place.", 1500, Check::Contains(vec!["232.3"])),
        c("M5", "math", "A building has 3 storeys of 3.2 m each plus a 1.1 m parapet. What is the total height in metres?", 900, Check::Contains(vec!["10.7"])),
        c("R1", "reasoning", "Every architect at the firm uses Revit. Maya is an architect at the firm. Does Maya use Revit? Answer yes or no first, then one short reason.", 800, Check::StartsYes),
        c("R2", "reasoning", "Sort these numbers in ascending order and output them comma-separated with nothing else: 42, 7, 19, 3, 25", 900, Check::Sorted("3,7,19,25,42")),
    ];
    v.push(Case { id: "T1", group: "tools", prompt: "What is the weather in Calgary right now? Use the tool.".into(), tools: Some(T_WEATHER), image: None, max_tok: 800, check: Check::Tool("get_weather", vec!["Calgary"]) });
    v.push(Case { id: "T2", group: "tools", prompt: "Use the calculate tool to compute 1234 * 5678.".into(), tools: Some(T_CALC), image: None, max_tok: 800, check: Check::Tool("calculate", vec!["1234", "5678"]) });
    v.push(Case { id: "T3", group: "tools", prompt: "Save a file called notes.txt containing the text: site visit Tuesday 9am.".into(), tools: Some(T_FILE), image: None, max_tok: 800, check: Check::Tool("write_file", vec!["notes.txt"]) });
    let rules = " Reply with one ```rust code block containing only the function (no main, no tests, std only).";
    v.push(c("C1", "coding", &format!("Write a Rust function `fn area(w: f64, h: f64) -> f64` that returns the area of a rectangle.{rules}"), 1500,
        Check::Rust("area", "assert!((area(4.2, 3.5) - 14.7).abs() < 1e-9); assert!(area(0.0, 5.0) == 0.0);")));
    v.push(c("C2", "coding", &format!("Write a Rust function `fn reverse_words(s: &str) -> String` that reverses the order of the space-separated words.{rules}"), 1500,
        Check::Rust("reverse_words", "assert_eq!(reverse_words(\"a b c\"), \"c b a\"); assert_eq!(reverse_words(\"one\"), \"one\");")));
    v.push(c("C3", "coding", &format!("Write a Rust function `fn parse_kn(s: &str) -> Option<f64>` that parses strings like \"12.5 kN\" (number, optional space, then kN) and returns the number, or None if it does not match.{rules}"), 2000,
        Check::Rust("parse_kn", "assert_eq!(parse_kn(\"12.5 kN\"), Some(12.5)); assert_eq!(parse_kn(\"3kN\"), Some(3.0)); assert_eq!(parse_kn(\"abc\"), None); assert_eq!(parse_kn(\"12.5 m\"), None);")));
    let q1 = format!("{}Question: What is the door code for Gate 7? Reply with the code and one short sentence.", haystack(1050, 525, "the door code for Gate 7 is 7341."));
    v.push(c("L1", "long-context", &q1, 1500, Check::Contains(vec!["7341"])));
    let q2 = format!("{}Question: What is the password for the site office? Reply with the password and one short sentence.", haystack(3500, 1200, "the password for the site office is maple-42."));
    v.push(c("L2", "long-context", &q2, 1500, Check::Contains(vec!["maple-42"])));
    // L3 (23 Nov 2026, single-seat max-context profile): INKEVAL_L3_ENTRIES=<n> adds a needle at ~30.7 tokens per entry
    // (16000 entries ~ 491K tokens), needle placed at 37% depth.
    if let Some(n) = std::env::var("INKEVAL_L3_ENTRIES").ok().and_then(|s| s.parse::<usize>().ok()) {
        let q3 = format!("{}Question: What is the serial number of the tower crane? Reply with the serial number and one short sentence.",
                         haystack(n, n * 37 / 100, "the serial number of the tower crane is TC-58219."));
        v.push(c("L3", "long-context", &q3, 1500, Check::Contains(vec!["TC-58219"])));
    }
    // INKEVAL_ONLY=L1,L3 runs just those cases.
    if let Ok(only) = std::env::var("INKEVAL_ONLY") {
        let keep: Vec<String> = only.split(',').map(|s| s.trim().to_string()).collect();
        v.retain(|x| keep.iter().any(|k| k == x.id));
    }
    let img = png(96, 96, &|x, y| if (24..72).contains(&x) && (24..72).contains(&y) { [220, 20, 20] } else { [255, 255, 255] });
    v.push(Case { id: "V1", group: "vision", prompt: "What shape is in the middle of this image, and what colour is it? Answer briefly.".into(), tools: None,
        image: Some(b64(&img)), max_tok: 900, check: Check::Contains(vec!["square", "red"]) });
    v
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 { eprintln!("usage: inkeval <host:port> <model> <out.jsonl> [rustc]"); std::process::exit(2); }
    let (addr, model, out_path) = (&a[1], &a[2], &a[3]);
    let rustc = a.get(4).cloned().unwrap_or_else(|| "rustc".into());
    let mut out = std::fs::File::create(out_path).expect("out file");
    let (mut pass, mut total) = (0, 0);
    let mut by_group: Vec<(String, u32, u32)> = Vec::new();
    for case in cases() {
        let content = match &case.image {
            Some(b) => format!("[{{\"type\":\"text\",\"text\":\"{}\"}},{{\"type\":\"image_url\",\"image_url\":{{\"url\":\"data:image/png;base64,{}\"}}}}]", esc(&case.prompt), b),
            None => format!("\"{}\"", esc(&case.prompt)),
        };
        let tools = case.tools.map(|t| format!(",\"tools\":{t}")).unwrap_or_default();
        let body = format!("{{\"model\":\"{model}\",\"temperature\":0,\"max_tokens\":{},\"messages\":[{{\"role\":\"user\",\"content\":{content}}}]{tools}}}", case.max_tok);
        let t0 = Instant::now();
        let resp = post(addr, &body, if case.id == "L3" { 5400 } else { 1200 });
        let secs = t0.elapsed().as_secs_f64();
        let (mut ok, mut note, mut answer, mut reasoning) = (false, String::new(), String::new(), String::new());
        let (mut ptok, mut ctok, mut finish) = (0, 0, String::new());
        match &resp {
            Err(e) => note = format!("request failed: {e}"),
            Ok(b) => {
                let mi = b.find("\"message\"").unwrap_or(0);
                answer = field(b, "content", mi).unwrap_or_default();
                reasoning = field(b, "reasoning", mi).or_else(|| field(b, "reasoning_content", mi)).unwrap_or_default();
                finish = field(b, "finish_reason", 0).unwrap_or_default();
                ptok = num(b, "prompt_tokens"); ctok = num(b, "completion_tokens");
                let low = answer.to_lowercase();
                match &case.check {
                    Check::Exact(w) => { ok = low.trim().trim_matches(|c: char| !c.is_alphanumeric()) == *w; }
                    Check::Contains(ws) => { let squash: String = low.chars().filter(|c| !c.is_whitespace()).collect();
                        ok = ws.iter().all(|w| low.contains(&w.to_lowercase()) || squash.contains(&w.to_lowercase())); if !ok && ws.iter().all(|w| reasoning.to_lowercase().contains(&w.to_lowercase())) { note = "found only in reasoning".into(); } }
                    Check::StartsYes => { ok = low.trim_start().starts_with("yes"); }
                    Check::JsonArray3 => { let t = answer.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
                        ok = t.starts_with('[') && t.ends_with(']') && t.matches('"').count() == 6; }
                    Check::Bullets3 => { let n = answer.lines().filter(|l| l.trim_start().starts_with("- ")).count();
                        let other = answer.lines().filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with("- ")).count(); ok = n == 3 && other == 0; }
                    Check::OneShortSentence => { let t = answer.trim(); let words = t.split_whitespace().count();
                        let ends = t.matches(|c| c == '.' || c == '!' || c == '?').count(); ok = words > 0 && words < 20 && ends <= 1; }
                    Check::Sorted(want) => { let t: String = answer.chars().filter(|c| !c.is_whitespace()).collect(); ok = t.trim_end_matches('.') == *want; }
                    Check::Tool(name, args) => {
                        let tc = b.find("\"tool_calls\"").unwrap_or(b.len());
                        let fname = if tc < b.len() { field(b, "name", tc).unwrap_or_default() } else { String::new() };
                        let fargs = if tc < b.len() { field(b, "arguments", tc).unwrap_or_default() } else { String::new() };
                        ok = fname == *name && args.iter().all(|x| fargs.contains(x));
                        note = format!("call {fname}({fargs})");
                    }
                    Check::Rust(name, tests) => match rust_block(&answer) {
                        Some(code) => { let (r, n) = run_rust(&rustc, &format!("{}_{}", case.id, name), &code, tests); ok = r; note = n; }
                        None => note = "no code block in the answer".into(),
                    },
                }
            }
        }
        total += 1; if ok { pass += 1; }
        match by_group.iter_mut().find(|g| g.0 == case.group) { Some(g) => { g.2 += 1; if ok { g.1 += 1; } } None => by_group.push((case.group.into(), ok as u32, 1)) }
        let line = format!("{{\"id\":\"{}\",\"group\":\"{}\",\"pass\":{},\"seconds\":{:.1},\"prompt_tokens\":{},\"completion_tokens\":{},\"finish\":\"{}\",\"note\":\"{}\",\"answer\":\"{}\",\"reasoning_head\":\"{}\"}}\n",
            case.id, case.group, ok, secs, ptok, ctok, esc(&finish), esc(&note), esc(&answer), esc(&reasoning.chars().take(600).collect::<String>()));
        out.write_all(line.as_bytes()).ok();
        println!("{:<3} {:<12} {:<4} {:6.1}s p={:<6} c={:<5} {} | {}", case.id, case.group, if ok { "PASS" } else { "FAIL" }, secs, ptok, ctok, note,
            answer.replace('\n', " ").chars().take(90).collect::<String>());
    }
    let groups: Vec<String> = by_group.iter().map(|(g, p, t)| format!("\"{g}\":[{p},{t}]")).collect();
    out.write_all(format!("{{\"summary\":true,\"pass\":{pass},\"total\":{total},\"groups\":{{{}}}}}\n", groups.join(",")).as_bytes()).ok();
    println!("TOTAL {pass}/{total}  {}", by_group.iter().map(|(g, p, t)| format!("{g} {p}/{t}")).collect::<Vec<_>>().join(", "));
}
