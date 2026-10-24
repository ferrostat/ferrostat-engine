use std::env;
use std::fs::File;
use std::io::{self, BufRead, Write};

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut interpreter = ferrostat_ado::interpreter::AdoInterpreter::new();

    // Batch mode: ferrostat --do file.do
    if args.len() >= 3 && args[1] == "--do" {
        let path = &args[2];
        match run_do_file(&mut interpreter, path) {
            Ok(_) => {}
            Err(e) => eprintln!("Error: {}", e),
        }
        return;
    }

    println!("Ferrostat clean-room engine v{}", env!("CARGO_PKG_VERSION"));
    println!("Type a command, or 'exit' to quit.");

    let stdin = io::stdin();

    loop {
        print!(". ");
        io::stdout().flush().unwrap();

        let mut line = String::new();
        if stdin.read_line(&mut line).is_err() {
            break;
        }

        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }

        if line.to_lowercase().starts_with("do ") {
            let path = line[3..].trim().trim_matches('"');
            match run_do_file(&mut interpreter, path) {
                Ok(_) => {}
                Err(e) => eprintln!("r({}); {}", 198, e),
            }
            continue;
        }

        match interpreter.run_line(line) {
            Ok(output) => {
                if output == "__EXIT__" {
                    break;
                }
                if !output.is_empty() {
                    println!("{}", output);
                }
            }
            Err(e) => {
                eprintln!("r({}); {}", 198, e);
            }
        }
    }
}

fn run_do_file(interpreter: &mut ferrostat_ado::interpreter::AdoInterpreter, path: &str) -> Result<(), String> {
    let file = File::open(path).map_err(|e| format!("cannot open {}: {}", path, e))?;
    let reader = io::BufReader::new(file);
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('*') || trimmed.starts_with("//") {
            continue;
        }
        match interpreter.run_line(trimmed) {
            Ok(output) => {
                if output == "__EXIT__" {
                    return Ok(());
                }
                if !output.is_empty() {
                    println!("{}", output);
                }
            }
            Err(e) => {
                eprintln!("r({}); {}", 198, e);
            }
        }
    }
    Ok(())
}
