// chatbench: real chat tokens per second against an OpenAI-compatible server (Fable, 22 Nov 2026). Rust, no dependencies.
// Three prompt kinds (prose, code, tool call), max_tokens 600, temperature 0.6,
// streamed, N chats at once. Each chat runs the three cases `rounds` times, starting at a different case.
//   chatbench <host:port> <model> <chats> <rounds> [think] [image=<png path>] [maxtok=<n>] [ctx=<text file>]
// ctx=: the file's text goes before every prompt (decode speed at long context; prefill shows as first_token_s).
// Per request: decode tok/s = (completion_tokens - 1) / (last chunk - first chunk); aggregate = all completion tokens /
// wall time from the first request sent to the last one finished.
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const TOOLS: &str = r#"[{"type":"function","function":{"name":"write_file","description":"Write a text file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]"#;
const CASES: [(&str, &str); 3] = [
    ("prose", "Explain in about 250 words why a timber-frame house needs a vapour barrier in a cold climate."),
    ("code", "Write a Python function that parses an ISO 8601 date string into year, month, day without libraries, with three asserts."),
    ("tool", "Save a short README for a project called Villa Clean that lists three sections: Overview, Layout, Furniture. Use the write_file tool."),
];

fn b64(d: &[u8]) -> String {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(d.len() * 4 / 3 + 4);
    for c in d.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        s.push(T[(n >> 18) as usize & 63] as char);
        s.push(T[(n >> 12) as usize & 63] as char);
        s.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        s.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    s
}

fn json_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + s.len() / 8);
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

struct Res { case: String, ok: bool, ttft: f64, total: f64, decode: f64, toks: u64, prompt: u64, finish: String, tool: bool }

fn num_after(s: &str, key: &str) -> Option<u64> {
    let i = s.rfind(key)? + key.len();
    let t: String = s[i..].chars().skip_while(|c| *c == ' ' || *c == ':').take_while(|c| c.is_ascii_digit()).collect();
    t.parse().ok()
}

fn request(addr: &str, body: &str, case: &str) -> Res {
    let mut r = Res { case: case.into(), ok: false, ttft: 0.0, total: 0.0, decode: 0.0, toks: 0, prompt: 0, finish: String::new(), tool: false };
    let t0 = Instant::now();
    let mut s = match TcpStream::connect(addr) { Ok(s) => s, Err(_) => return r };
    s.set_read_timeout(Some(Duration::from_secs(900))).ok();
    let req = format!("POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}", addr, body.len(), body);
    if s.write_all(req.as_bytes()).is_err() { return r; }
    let mut rd = BufReader::new(s);
    let mut line = String::new();
    let (mut first, mut last): (Option<Instant>, Instant) = (None, t0);
    let mut status_ok = false;
    loop {
        line.clear();
        match rd.read_line(&mut line) { Ok(0) | Err(_) => break, _ => {} }
        if line.starts_with("HTTP/1.1 200") { status_ok = true; }
        if !line.starts_with("data:") { continue; }
        if line.contains("[DONE]") { break; }
        if line.contains("\"delta\"") && (line.contains("\"content\":\"") || line.contains("\"reasoning") || line.contains("\"tool_calls\"")) {
            if first.is_none() { first = Some(Instant::now()); }
            last = Instant::now();
        }
        if line.contains("\"tool_calls\"") { r.tool = true; }
        if let Some(i) = line.find("\"finish_reason\":\"") { r.finish = line[i + 17..].split('"').next().unwrap_or("").to_string(); }
        if line.contains("\"usage\":{") {
            r.toks = num_after(&line, "\"completion_tokens\"").unwrap_or(0);
            r.prompt = num_after(&line, "\"prompt_tokens\"").unwrap_or(0);
        }
    }
    let mut rest = String::new();
    rd.read_to_string(&mut rest).ok();
    r.total = t0.elapsed().as_secs_f64();
    if let Some(f) = first {
        r.ttft = f.duration_since(t0).as_secs_f64();
        let d = last.duration_since(f).as_secs_f64();
        if d > 0.0 && r.toks > 1 { r.decode = (r.toks - 1) as f64 / d; }
    }
    r.ok = status_ok && r.toks > 0;
    r
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 { eprintln!("usage: chatbench <host:port> <model> <chats> <rounds> [think] [image=<png>] [maxtok=<n>]"); std::process::exit(2); }
    let (addr, model) = (a[1].clone(), a[2].clone());
    let (chats, rounds): (usize, usize) = (a[3].parse().unwrap(), a[4].parse().unwrap());
    let think = a.iter().any(|x| x == "think");
    let maxtok: u64 = a.iter().find_map(|x| x.strip_prefix("maxtok=")).map(|x| x.parse().unwrap()).unwrap_or(600);
    let extra = if think { String::new() } else { ",\"chat_template_kwargs\":{\"enable_thinking\":false}".to_string() };
    if let Some(img) = a.iter().find_map(|x| x.strip_prefix("image=")) {
        let data = std::fs::read(img).expect("image");
        let body = format!("{{\"model\":\"{}\",\"stream\":true,\"stream_options\":{{\"include_usage\":true}},\"max_tokens\":{},\"temperature\":0.6{},\"messages\":[{{\"role\":\"user\",\"content\":[{{\"type\":\"text\",\"text\":\"Describe this picture in two sentences.\"}},{{\"type\":\"image_url\",\"image_url\":{{\"url\":\"data:image/png;base64,{}\"}}}}]}}]}}", model, maxtok.min(120), extra, b64(&data));
        for i in 0..rounds {
            let r = request(&addr, &body, "picture");
            println!("{{\"case\":\"picture\",\"run\":{},\"ok\":{},\"seconds\":{:.2},\"first_token_s\":{:.2},\"completion_tokens\":{},\"prompt_tokens\":{},\"decode_tok_s\":{:.1}}}", i, r.ok, r.total, r.ttft, r.toks, r.prompt, r.decode);
        }
        return;
    }
    let ctx: String = a.iter().find_map(|x| x.strip_prefix("ctx="))
        .map(|p| format!("{}\\n\\nTask: ", json_escape(&std::fs::read_to_string(p).expect("ctx file"))))
        .unwrap_or_default();
    // edit=<file>: every request is a copy-heavy edit of that file (the Atelier kind of work): return the whole
    // file with one comment line added at the top (dual-drafter test, 23 Nov 2026).
    let edit: String = a.iter().find_map(|x| x.strip_prefix("edit="))
        .map(|p| format!("Here is a source file:\\n\\n{}\\n\\nReturn the complete file unchanged except for one new first line: a comment saying reviewed. Reply with only the file in one code block.", json_escape(&std::fs::read_to_string(p).expect("edit file"))))
        .unwrap_or_default();
    let out: Arc<Mutex<Vec<Res>>> = Arc::new(Mutex::new(vec![]));
    let t0 = Instant::now();
    let hs: Vec<_> = (0..chats).map(|c| {
        // Inkling's template reads reasoning_effort (default high), not enable_thinking: an edit is answered
        // without reasoning so the output is the copied file.
        let extra = if edit.is_empty() { extra.clone() } else { ",\"chat_template_kwargs\":{\"reasoning_effort\":\"none\"}".to_string() };
        let (addr, model, out, ctx, edit) = (addr.clone(), model.clone(), out.clone(), ctx.clone(), edit.clone());
        std::thread::spawn(move || {
            for i in 0..rounds * 3 {
                let (name, prompt) = if edit.is_empty() { CASES[(c + i) % 3] } else { ("edit", edit.as_str()) };
                let tools = if name == "tool" { format!(",\"tools\":{}", TOOLS) } else { String::new() };
                let body = format!("{{\"model\":\"{}\",\"stream\":true,\"stream_options\":{{\"include_usage\":true}},\"max_tokens\":{},\"temperature\":0.6{}{},\"messages\":[{{\"role\":\"user\",\"content\":\"{}{}\"}}]}}", model, maxtok, extra, tools, ctx, prompt);
                let r = request(&addr, &body, name);
                out.lock().unwrap().push(r);
            }
        })
    }).collect();
    for h in hs { h.join().unwrap(); }
    let wall = t0.elapsed().as_secs_f64();
    let v = out.lock().unwrap();
    let (mut toks, mut okn) = (0u64, 0usize);
    for r in v.iter() {
        println!("{{\"case\":\"{}\",\"ok\":{},\"seconds\":{:.2},\"first_token_s\":{:.2},\"completion_tokens\":{},\"prompt_tokens\":{},\"decode_tok_s\":{:.1},\"tok_s_end_to_end\":{:.1},\"finish\":\"{}\",\"tool_call\":{}}}", r.case, r.ok, r.total, r.ttft, r.toks, r.prompt, r.decode, if r.total > 0.0 { r.toks as f64 / r.total } else { 0.0 }, r.finish, r.tool);
        if r.ok { toks += r.toks; okn += 1; }
    }
    let mut d: Vec<f64> = v.iter().filter(|r| r.ok && r.decode > 0.0).map(|r| r.decode).collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = if d.is_empty() { 0.0 } else { d[d.len() / 2] };
    let by = |c: &str| { let mut x: Vec<f64> = v.iter().filter(|r| r.ok && r.case == c && r.decode > 0.0).map(|r| r.decode).collect(); x.sort_by(|a, b| a.partial_cmp(b).unwrap()); if x.is_empty() { 0.0 } else { x[x.len() / 2] } };
    let e2e: Vec<f64> = v.iter().filter(|r| r.ok).map(|r| r.toks as f64 / r.total).collect();
    let e2e_mean = if e2e.is_empty() { 0.0 } else { e2e.iter().sum::<f64>() / e2e.len() as f64 };
    println!("{{\"summary\":true,\"mean_tok_s_end_to_end_per_chat\":{:.1},\"sum_over_chats\":{:.1},", e2e_mean, e2e_mean * chats as f64);
    println!("{{\"summary2\":true,\"chats\":{},\"requests\":{},\"ok\":{},\"wall_s\":{:.2},\"completion_tokens\":{},\"aggregate_tok_s\":{:.1},\"median_decode_tok_s_per_chat\":{:.1},\"prose\":{:.1},\"code\":{:.1},\"tool\":{:.1},\"edit\":{:.1}}}",
        chats, v.len(), okn, wall, toks, toks as f64 / wall, med, by("prose"), by("code"), by("tool"), by("edit"));
}
