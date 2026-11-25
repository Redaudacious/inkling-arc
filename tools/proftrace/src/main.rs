// proftrace: attribute torch-profiler ops to the source line that issued them (inkling-arc, 23 Nov 2026).
//
//   proftrace <trace.json[.gz]> [ops=aten::copy_,aten::mm,...] [match=inkling2,speculator,dspark] [top=40]
//
// Reads a chrome trace written by torch.profiler with with_stack=True. For every CPU op whose name is in `ops`, finds
// the innermost enclosing python_function event on the same thread whose frame name contains one of the `match`
// strings (our engine files), and sums count and CPU time per (op, frame). With the per-op XPU totals of the same
// window this tells which line of the engine to change to remove a launch.
use flate2::read::GzDecoder;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};

struct Ev {
    ts: f64,
    end: f64,
    name: String,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: proftrace <trace.json[.gz]> [ops=a,b] [match=x,y] [top=N]");
        std::process::exit(2);
    }
    let opt = |k: &str, d: &str| -> String {
        args.iter().find_map(|a| a.strip_prefix(&format!("{k}=")).map(str::to_string)).unwrap_or_else(|| d.to_string())
    };
    let ops: Vec<String> = opt("ops", "aten::copy_,aten::mm,aten::addmm,aten::bmm,_xpu_C::int8_scaled_mm,aten::fill_")
        .split(',').map(str::to_string).collect();
    let matches: Vec<String> = opt("match", "inkling2,speculator,dspark,model_runner")
        .split(',').map(str::to_string).collect();
    let top: usize = opt("top", "40").parse().unwrap_or(40);

    let path = &args[1];
    let f = BufReader::new(File::open(path).expect("open trace"));
    let mut text = String::new();
    if path.ends_with(".gz") {
        GzDecoder::new(f).read_to_string(&mut text).expect("gunzip");
    } else {
        BufReader::new(f).read_to_string(&mut text).expect("read");
    }
    let root: Value = serde_json::from_str(&text).expect("json");
    drop(text);
    let evs = root.get("traceEvents").and_then(Value::as_array).expect("traceEvents");

    // python frames per thread, and the target ops
    let mut frames: HashMap<String, Vec<Ev>> = HashMap::new();
    let mut targets: Vec<(String, Ev)> = Vec::new();
    for e in evs {
        if e.get("ph").and_then(Value::as_str) != Some("X") {
            continue;
        }
        let cat = e.get("cat").and_then(Value::as_str).unwrap_or("");
        let name = e.get("name").and_then(Value::as_str).unwrap_or("");
        let tid = format!("{}:{}", e.get("pid").map(|v| v.to_string()).unwrap_or_default(),
                          e.get("tid").map(|v| v.to_string()).unwrap_or_default());
        let ts = e.get("ts").and_then(Value::as_f64).unwrap_or(0.0);
        let dur = e.get("dur").and_then(Value::as_f64).unwrap_or(0.0);
        let ev = Ev { ts, end: ts + dur, name: name.to_string() };
        if cat == "python_function" {
            if matches.iter().any(|m| name.contains(m.as_str())) {
                frames.entry(tid).or_default().push(ev);
            }
        } else if cat == "cpu_op" && ops.iter().any(|o| o == name) {
            targets.push((tid, ev));
        }
    }
    for v in frames.values_mut() {
        v.sort_by(|a, b| a.ts.partial_cmp(&b.ts).unwrap());
    }

    // innermost matching frame = the latest-starting one that still contains the op
    let mut agg: HashMap<(String, String), (u64, f64)> = HashMap::new();
    let mut per_op: HashMap<String, u64> = HashMap::new();
    for (tid, op) in &targets {
        *per_op.entry(op.name.clone()).or_default() += 1;
        let frame = frames.get(tid).and_then(|fs| {
            let hi = fs.partition_point(|f| f.ts <= op.ts);
            fs[..hi].iter().rev().take(4000).find(|f| f.end >= op.end).map(|f| f.name.clone())
        }).unwrap_or_else(|| "(no engine frame)".to_string());
        let a = agg.entry((op.name.clone(), frame)).or_default();
        a.0 += 1;
        a.1 += op.end - op.ts;
    }
    let mut rows: Vec<_> = agg.into_iter().collect();
    rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    println!("ops in window: {:?}", per_op);
    println!("{:>7} {:>10}  {:<24} frame", "calls", "cpu_us", "op");
    for ((op, frame), (n, us)) in rows.into_iter().take(top) {
        println!("{n:>7} {us:>10.0}  {op:<24} {frame}");
    }
}
