use ferrostat_engine::engine::Engine;
use ferrostat_parser::parser::{parse_line, Command};

/// Simple ado-script interpreter.
/// Supports:
///   - program define name
///   - end
///   - version N
///   - syntax [<varlist>] [if] [in] [, options]
///   - local macro = expression
///   - display
///   - calling built-in commands
pub struct AdoInterpreter {
    engine: Engine,
    programs: std::collections::HashMap<String, Vec<String>>,
    in_program: bool,
    current_program: String,
    current_body: Vec<String>,
}

impl AdoInterpreter {
    pub fn new() -> Self {
        Self {
            engine: Engine::new(),
            programs: std::collections::HashMap::new(),
            in_program: false,
            current_program: String::new(),
            current_body: Vec::new(),
        }
    }

    pub fn run_line(&mut self, line: &str) -> Result<String, String> {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('*') || trimmed.starts_with("//") {
            return Ok(String::new());
        }

        if self.in_program {
            if trimmed.eq_ignore_ascii_case("end") {
                self.in_program = false;
                let body = std::mem::take(&mut self.current_body);
                let name = std::mem::take(&mut self.current_program);
                self.programs.insert(name.clone(), body);
                return Ok(format!("program {} defined", name));
            }
            self.current_body.push(trimmed.to_string());
            return Ok(String::new());
        }

        // Check for program define
        if trimmed.to_lowercase().starts_with("program define ") {
            let name = trimmed[15..].trim().to_string();
            self.in_program = true;
            self.current_program = name;
            return Ok(String::new());
        }

        // Check for user-defined programs
        let cmd = match parse_line(trimmed) {
            Ok(c) => c,
            Err(e) => return Err(format!("parse error: {}", e)),
        };
        if let Some(body) = self.programs.get(&cmd.verb).cloned() {
            return self.run_program(&cmd.verb, &cmd, &body);
        }

        // Pass to engine
        self.engine.execute(&cmd)
    }

    fn run_program(&mut self, _name: &str, _cmd: &Command, body: &[String]) -> Result<String, String> {
        let mut output = Vec::new();
        for line in body {
            let line = line.trim();
            if line.is_empty() || line.starts_with('*') {
                continue;
            }
            let cmd = match parse_line(line) {
                Ok(c) => c,
                Err(e) => return Err(format!("parse error: {}", e)),
            };
            match self.engine.execute(&cmd) {
                Ok(o) => {
                    if o == "__EXIT__" {
                        return Ok("__EXIT__".into());
                    }
                    if !o.is_empty() {
                        output.push(o);
                    }
                }
                Err(e) => return Err(format!("error in program: {}", e)),
            }
        }
        Ok(output.join("\n"))
    }

    pub fn engine(&self) -> &Engine { &self.engine }
    pub fn engine_mut(&mut self) -> &mut Engine { &mut self.engine }
}
