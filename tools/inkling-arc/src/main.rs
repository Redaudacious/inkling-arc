// inkling-arc: Inkling 2 Mini on one Intel Arc Pro B70 (32 GB), lossless NVFP4/MXFP8 experts, K8/V4 global KV cache.
// One command: `./inkling-arc install` = doctor -> build -> fetch -> pack -> serve (multi) -> test quick.
// Rust, std only. Every heavy step runs in a container (podman, else docker); nothing runs on the CPU that belongs on the GPU.
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{exit, Command, Stdio};
use std::time::{Duration, Instant};

// The checkpoint: any Hugging Face repo in Thinking Machines' NVFP4/MXFP8 layout. INKLING_ARC_MODEL picks another one,
// e.g. arcitech-psp/Inkling-2-Mini-Heretic-NVFP4-XPU (DavidAU's fine-tune, converted with tools/inkconv).
const DEFAULT_MODEL: &str = "thinkingmachines/Inkling-2-Mini";
const BASE_IMAGE: &str = "localhost/vllm-xpu:b70-inkling2";
const BUILDER_IMAGE: &str = "localhost/inkling-arc-builder:2026.1.1";
const FINAL_IMAGE: &str = "localhost/inkling-arc:latest";
const CONTAINER: &str = "inkling-arc";

// long_cap: while other requests exist, a long prompt reads at most this many tokens per step (0 = off). Measured
// 24 Nov on the B70: with 2,560 / 2,048 another chat's new request waits 1.75 s instead of 7.8-9.2 s while a 32K prompt
// is read, and the long prompt itself is not slower.
struct Profile { name: &'static str, ctx: u32, seqs: u32, util: &'static str, batched: u32, long_cap: u32, what: &'static str }
const PROFILES: [Profile; 2] = [
    Profile { name: "multi", ctx: 131_072, seqs: 4, util: "0.95", batched: 2560, long_cap: 2048, what: "4 chats x 131,072 tokens (Atelier / DSH)" },
    Profile { name: "single", ctx: 524_288, seqs: 1, util: "0.97", batched: 2048, long_cap: 0, what: "1 chat x 524,288 tokens" },
];

struct Cfg { root: PathBuf, data: PathBuf, engine: String, port: u16, render: Option<String> }

fn die(msg: &str) -> ! { eprintln!("inkling-arc: {msg}"); exit(1) }
fn say(msg: &str) { println!("== {msg}"); }

fn run(cmd: &mut Command) {
    let shown = format!("{cmd:?}");
    match cmd.status() {
        Ok(s) if s.success() => {}
        Ok(s) => die(&format!("failed ({s}): {}", &shown[..shown.len().min(300)])),
        Err(e) => die(&format!("cannot start {}: {e}", &shown[..shown.len().min(120)])),
    }
}
fn ok(cmd: &mut Command) -> bool { cmd.stdout(Stdio::null()).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false) }
fn out(cmd: &mut Command) -> String { cmd.output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default() }

fn engine() -> String {
    for e in ["podman", "docker"] { if ok(Command::new(e).arg("--version")) { return e.into(); } }
    die("needs podman or docker")
}

// The B70's render node: Intel vendor 0x8086 with the Arc Pro B70 device id 0xe223 (BMG-G31), by-path name for stability.
fn render_node() -> Option<String> {
    let dir = fs::read_dir("/dev/dri/by-path").ok()?;
    let mut found = None;
    for e in dir.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if !name.ends_with("-render") { continue; }
        let pci = name.trim_start_matches("pci-").trim_end_matches("-render");
        let vendor = fs::read_to_string(format!("/sys/bus/pci/devices/{pci}/vendor")).unwrap_or_default();
        let device = fs::read_to_string(format!("/sys/bus/pci/devices/{pci}/device")).unwrap_or_default();
        if vendor.trim() == "0x8086" && (device.trim() == "0xe223" || env::var("INKLING_ARC_ANY_INTEL").is_ok()) {
            found = Some(format!("/dev/dri/by-path/{name}"));
        }
    }
    found
}

fn gpu_args(cfg: &Cfg) -> Vec<String> {
    let r = cfg.render.clone().unwrap_or_else(|| die("no Arc Pro B70 render node found (INKLING_ARC_ANY_INTEL=1 to try another Intel GPU)"));
    let real_r = fs::canonicalize(&r).map(|p| p.display().to_string()).unwrap_or(r.clone());
    let c = r.replace("-render", "-card");
    let real_c = fs::canonicalize(&c).map(|p| p.display().to_string()).unwrap_or(c);
    let mut a = vec!["--device".into(), real_c, "--device".into(), real_r, "-v".into(), "/dev/dri/by-path:/dev/dri/by-path".into(), "--ipc=host".into()];
    if cfg.engine == "podman" { a.push("--group-add".into()); a.push("keep-groups".into()); }
    a
}

fn image_exists(cfg: &Cfg, tag: &str) -> bool { ok(Command::new(&cfg.engine).args(["image", "inspect", tag])) }

fn doctor(cfg: &Cfg) {
    say("doctor");
    println!("engine      {}", cfg.engine);
    println!("gpu         {}", cfg.render.clone().unwrap_or_else(|| "NOT FOUND (needs an Intel Arc Pro B70)".into()));
    let df = out(Command::new("df").args(["-BG", "--output=avail"]).arg(&cfg.data));
    println!("free disk   {} (needs ~70 GB: images 25, checkpoint 20, packs 15)", df.lines().last().unwrap_or("?").trim());
    for (tag, what) in [(BASE_IMAGE, "base image"), (BUILDER_IMAGE, "builder image"), (FINAL_IMAGE, "serving image")] {
        println!("{what:<11} {}", if image_exists(cfg, tag) { "present" } else { "missing (inkling-arc build)" });
    }
    println!("checkpoint  {}", if cfg.data.join("model/model-00001-of-00001.safetensors").exists() { "present" } else { "missing (inkling-arc fetch)" });
    println!("packs       {}", if cfg.data.join("packs/L23.w2.mx8.nvf4").exists() { "present" } else { "missing (inkling-arc pack)" });
    if cfg.render.is_none() { exit(2); }
}

fn build(cfg: &Cfg, skip_base: bool) {
    let r = &cfg.root;
    if !image_exists(cfg, BASE_IMAGE) {
        if skip_base { die("base image missing and --skip-base given"); }
        say("base image (vLLM-XPU Inkling fork on the B70 kit; long, CPU compile only)");
        run(Command::new("bash").arg(r.join("image/build-base-image.sh")).env("INKLING_ARC_ENGINE", &cfg.engine));
    }
    if !image_exists(cfg, BUILDER_IMAGE) {
        say("builder image (oneAPI DPC++ 2026.1.1 on the base environment)");
        run(Command::new(&cfg.engine).args(["build", "--format", "docker", "-t", BUILDER_IMAGE, "-f"]).arg(r.join("image/Containerfile.builder")).arg(r.join("image")));
    }
    say("kernels (experts, dense, attention + oneDNN, pack library), compiled for the B70");
    fs::create_dir_all(r.join("build")).ok();
    run(Command::new(&cfg.engine).args(["run", "--rm", "-v"]).arg(format!("{}:/work", r.display())).args(["--entrypoint", "bash", BUILDER_IMAGE, "/work/kernels/build-all.sh"]));
    say("host tools (nvf4 packer, inkeval, chatbench, proftrace, inkconv)");
    run(Command::new(&cfg.engine).args(["run", "--rm", "-v"]).arg(format!("{}:/work", r.display()))
        .args(["-w", "/work", "docker.io/library/rust:1-bookworm", "bash", "-c",
               "for t in nvf4 inkeval chatbench proftrace inkconv; do cargo build --release --quiet --manifest-path tools/$t/Cargo.toml --target-dir build/rust || exit 1; done"]));
    say("serving image");
    run(Command::new(&cfg.engine).args(["build", "--format", "docker", "--build-arg"]).arg(format!("BASE={BASE_IMAGE}"))
        .args(["-t", FINAL_IMAGE, "-f"]).arg(r.join("image/Containerfile")).arg(r));
}

fn fetch(cfg: &Cfg) {
    let repo = env::var("INKLING_ARC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    say(&format!("checkpoint from Hugging Face: {repo} (~20 GB)"));
    let dir = cfg.data.join("model");
    fs::create_dir_all(&dir).ok();
    let rev = env::var("INKLING_ARC_REVISION").unwrap_or_else(|_| "main".into());
    run(Command::new(&cfg.engine).args(["run", "--rm", "-v"]).arg(format!("{}:/model", dir.display()))
        .args(["-e", "HF_HUB_ENABLE_HF_TRANSFER=0", "--entrypoint", "hf", FINAL_IMAGE, "download", &repo, "--revision", &rev, "--local-dir", "/model"]));
}

fn pack(cfg: &Cfg) {
    say("lossless expert packs on the GPU (codes unchanged, DPAS tile order; ~1-6 min)");
    let p = cfg.data.join("packs");
    fs::create_dir_all(&p).ok();
    let mut c = Command::new(&cfg.engine);
    c.args(["run", "--rm"]).args(gpu_args(cfg))
        .arg("-v").arg(format!("{}:/model:ro", cfg.data.join("model").display()))
        .arg("-v").arg(format!("{}:/packs", p.display()))
        .args(["--entrypoint", "nvf4", FINAL_IMAGE, "convert", "/model/model-00001-of-00001.safetensors", "/packs", "0-23", "s8"]);
    run(&mut c);
}

fn stop(cfg: &Cfg) { let _ = Command::new(&cfg.engine).args(["rm", "-f", CONTAINER]).stdout(Stdio::null()).stderr(Stdio::null()).status(); }

fn serve(cfg: &Cfg, profile: &str) {
    let pr = PROFILES.iter().find(|p| p.name == profile).unwrap_or_else(|| die("profile is multi or single"));
    for need in ["model/model-00001-of-00001.safetensors", "packs/L23.w2.mx8.nvf4"] {
        if !cfg.data.join(need).exists() { die(&format!("missing {need}: run inkling-arc fetch / pack")); }
    }
    stop(cfg);
    // compiled Triton kernels and the GPU driver's compiler cache outlive the container: a restart is ~25% faster
    // and the first long prompt after it skips ~3.7 s of kernel compiles (measured 24 Nov 2026)
    let (tc, nc) = (cfg.data.join("cache/triton"), cfg.data.join("cache/neo"));
    for d in [&tc, &nc] { let _ = std::fs::create_dir_all(d); }
    say(&format!("serve {} = {} on port {}", pr.name, pr.what, cfg.port));
    let spec = r#"{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}"#;
    let mut c = Command::new(&cfg.engine);
    c.args(["run", "-d", "--name", CONTAINER, "--restart=no", "--memory=16g", "--ulimit", "core=0"]).args(gpu_args(cfg))
        .args(["-p", &format!("{}:8000", cfg.port)])
        .arg("-v").arg(format!("{}:/models/inkling-2-mini:ro", cfg.data.join("model").display()))
        .arg("-v").arg(format!("{}:/packs:ro", cfg.data.join("packs").display()))
        .arg("-v").arg(format!("{}:/root/.triton/cache", tc.display()))
        .arg("-v").arg(format!("{}:/root/.cache/neo_compiler_cache", nc.display()))
        .args([FINAL_IMAGE, "serve", "/models/inkling-2-mini", "--served-model-name", "inkling-2-mini", "--trust-remote-code",
               "--host", "0.0.0.0", "--port", "8000", "--tensor-parallel-size", "1"])
        .args(["--max-model-len", &pr.ctx.to_string(), "--max-num-seqs", &pr.seqs.to_string(),
               "--max-num-batched-tokens", &pr.batched.to_string(), "--gpu-memory-utilization", pr.util, "--enable-prefix-caching",
               "--reasoning-parser", "inkling", "--tool-call-parser", "inkling", "--enable-auto-tool-choice",
               "--speculative-config", spec, "--compilation-config", r#"{"cudagraph_mode":"FULL_DECODE_ONLY"}"#]);
    if pr.long_cap > 0 { c.args(["--long-prefill-token-threshold", &pr.long_cap.to_string()]); }
    if let Ok(alias) = env::var("INKLING_ARC_ALIASES") { for a in alias.split(',') { c.args(["--served-model-name", a]); } }
    run(c.stdout(Stdio::null()));
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(900) {
        if http_get(cfg.port, "/v1/models").map(|b| b.contains("inkling-2-mini")).unwrap_or(false) {
            say(&format!("ready after {} s: http://127.0.0.1:{}/v1 (model inkling-2-mini)", t0.elapsed().as_secs(), cfg.port));
            return;
        }
        let state = out(Command::new(&cfg.engine).args(["inspect", "-f", "{{.State.Status}}", CONTAINER]));
        if state == "exited" { let _ = Command::new(&cfg.engine).args(["logs", "--tail", "40", CONTAINER]).status(); die("server exited"); }
        std::thread::sleep(Duration::from_secs(5));
    }
    die("server not ready after 15 min");
}

fn http_get(port: u16, path: &str) -> Option<String> {
    let mut s = TcpStream::connect(("127.0.0.1", port)).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    write!(s, "GET {path} HTTP/1.0\r\nHost: localhost\r\n\r\n").ok()?;
    let mut b = String::new();
    s.read_to_string(&mut b).ok()?;
    Some(b)
}

fn test(cfg: &Cfg, level: &str) {
    let r = &cfg.root;
    say("kernel tests on the GPU (attention vs reference, far-key prefill vs reference, experts vs f64 reference)");
    for t in ["tests/kernels/test_kvq_attention.py", "tests/kernels/test_far_prefill.py"] {
        let mut c = Command::new(&cfg.engine);
        c.args(["run", "--rm"]).args(gpu_args(cfg)).arg("-v").arg(format!("{}:/work:ro", r.display()))
            .args(["-e", "PREFILL=0", "-e", "PASTMAX=0", "-e", "SPEED=0", "--entrypoint", "python3", FINAL_IMAGE]).arg(format!("/work/{t}"));
        run(&mut c);
    }
    say("stop words only in answer text (real tokenizer, CPU): reasoning ignored, block boundary, tool JSON, raw prompts");
    run(Command::new(&cfg.engine).args(["run", "--rm", "-v"]).arg(format!("{}:/work:ro", r.display()))
        .arg("-v").arg(format!("{}:/model:ro", cfg.data.join("model").display()))
        .args(["--entrypoint", "python3", FINAL_IMAGE, "/work/tests/test_stop_gate.py", "/model"]));
    say("model eval (20 cases: instructions, maths, reasoning, tools, Rust code compiled and run, 31K + 107K needles, vision)");
    // inside the Rust image: the coding cases compile and run the model's Rust answers with rustc
    let mut c = rust_tool(cfg, "inkeval");
    c.args([&format!("127.0.0.1:{}", cfg.port), "inkling-2-mini", &format!("/data/eval-{level}.jsonl")]);
    if level == "full" { c.env("INKEVAL_L3_ENTRIES", "16000"); }
    run(&mut c);
}

fn rust_tool(cfg: &Cfg, tool: &str) -> Command {
    let mut c = Command::new(&cfg.engine);
    c.args(["run", "--rm", "--network", "host", "-e", "INKEVAL_L3_ENTRIES"])
        .arg("-v").arg(format!("{}:/work:ro", cfg.root.display()))
        .arg("-v").arg(format!("{}:/data", cfg.data.display()))
        .args(["docker.io/library/rust:1-bookworm", &format!("/work/build/rust/release/{tool}")]);
    c
}

fn bench(cfg: &Cfg) {
    say("speed: 1 chat and 4 chats (prose, code, tool call; 600 tokens each)");
    let addr = format!("127.0.0.1:{}", cfg.port);
    run(rust_tool(cfg, "chatbench").args([&addr, "inkling-2-mini", "1", "2"]));
    run(rust_tool(cfg, "chatbench").args([&addr, "inkling-2-mini", "4", "2"]));
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let root = env::var("INKLING_ARC_ROOT").map(PathBuf::from).unwrap_or_else(|_| env::current_dir().unwrap());
    let data = env::var("INKLING_ARC_DATA").map(PathBuf::from).unwrap_or_else(|_| root.join("data"));
    fs::create_dir_all(&data).ok();
    let port = env::var("INKLING_ARC_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8000);
    let cfg = Cfg { root, data, engine: engine(), port, render: render_node() };
    let a1 = args.get(0).map(String::as_str).unwrap_or("help");
    let a2 = args.get(1).map(String::as_str);
    match a1 {
        "doctor" => doctor(&cfg),
        "build" => build(&cfg, args.iter().any(|a| a == "--skip-base")),
        "fetch" => fetch(&cfg),
        "pack" => pack(&cfg),
        "serve" => serve(&cfg, a2.unwrap_or("multi")),
        "stop" => stop(&cfg),
        "test" => test(&cfg, a2.unwrap_or("quick")),
        "bench" => bench(&cfg),
        "install" => {
            doctor(&cfg);
            build(&cfg, false);
            if !cfg.data.join("model/model-00001-of-00001.safetensors").exists() { fetch(&cfg); }
            if !cfg.data.join("packs/L23.w2.mx8.nvf4").exists() { pack(&cfg); }
            serve(&cfg, a2.unwrap_or("multi"));
            test(&cfg, "quick");
        }
        _ => {
            println!("inkling-arc - Inkling 2 Mini on one Intel Arc Pro B70\n");
            println!("  install [multi|single]  everything: doctor, build, fetch, pack, serve, test");
            println!("  doctor                  check GPU, engine, disk, images, files");
            println!("  build [--skip-base]     images + kernels + tools");
            println!("  fetch | pack            checkpoint from Hugging Face | lossless expert packs on the GPU");
            println!("  serve multi|single      {} | {}", PROFILES[0].what, PROFILES[1].what);
            println!("  stop | test quick|full | bench");
            println!("\nenv: INKLING_ARC_DATA (default ./data), INKLING_ARC_PORT (8000), INKLING_ARC_MODEL ({DEFAULT_MODEL}),");
            println!("     INKLING_ARC_REVISION (main), INKLING_ARC_ALIASES");
            let _ = Path::new("");
        }
    }
}
