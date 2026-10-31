//! Network access: a small blocking HTTP(S) client used by `copy`, `use`,
//! `import ... using http://...`, the package commands and `update`.
//!
//! The connection settings (`set httpproxy`, `httpproxyhost`, ...,
//! `timeout1`, `timeout2`) are kept in a process-wide configuration that the
//! interpreter updates whenever one of them is set. The browser build has
//! no network layer: every fetch fails with a clear message (r(631)).

use std::io;
use std::sync::Mutex;

/// Connection settings (the `netio` group of `set`).
#[derive(Clone, Debug, PartialEq)]
pub struct NetConfig {
    pub proxy_on: bool,
    pub proxy_host: String,
    pub proxy_port: u16,
    pub proxy_auth: bool,
    pub proxy_user: String,
    pub proxy_pw: String,
    /// seconds to wait for a connection
    pub timeout1: u64,
    /// seconds to wait for data once connected
    pub timeout2: u64,
}

impl Default for NetConfig {
    fn default() -> Self {
        NetConfig { proxy_on: false, proxy_host: String::new(), proxy_port: 80, proxy_auth: false, proxy_user: String::new(), proxy_pw: String::new(), timeout1: 30, timeout2: 180 }
    }
}

static CONFIG: Mutex<Option<NetConfig>> = Mutex::new(None);

/// Replace the connection settings.
pub fn configure(cfg: NetConfig) {
    if let Ok(mut g) = CONFIG.lock() {
        *g = Some(cfg);
    }
}

/// The current connection settings.
pub fn config() -> NetConfig {
    CONFIG.lock().ok().and_then(|g| g.clone()).unwrap_or_default()
}

/// Whether `s` is an http:// or https:// address.
pub fn is_url(s: &str) -> bool {
    let l = s.trim_start().get(..8).unwrap_or(s).to_ascii_lowercase();
    l.starts_with("http://") || l.starts_with("https://")
}

/// Whether this build can reach the network.
pub fn available() -> bool {
    cfg!(not(target_arch = "wasm32"))
}

/// Error text used when the browser build is asked for a URL.
pub const NO_NETWORK: &str = "network access is not available in the browser build";

/// User-Agent sent with every request.
pub fn user_agent() -> String {
    format!("ferrostat/{}", env!("CARGO_PKG_VERSION"))
}

/// The proxy address implied by `cfg`, if the proxy is switched on.
pub fn proxy_url(cfg: &NetConfig) -> Option<String> {
    if !cfg.proxy_on || cfg.proxy_host.trim().is_empty() {
        return None;
    }
    let host = cfg.proxy_host.trim();
    let host = host.strip_prefix("http://").unwrap_or(host).trim_end_matches('/');
    let auth = if cfg.proxy_auth && !cfg.proxy_user.is_empty() { format!("{}:{}@", pct(&cfg.proxy_user), pct(&cfg.proxy_pw)) } else { String::new() };
    let port = if host.contains(':') { String::new() } else { format!(":{}", cfg.proxy_port) };
    Some(format!("http://{auth}{host}{port}"))
}

fn pct(s: &str) -> String {
    s.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-._~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

/// Download `url` (http or https) and return its body.
#[cfg(not(target_arch = "wasm32"))]
pub fn fetch_url(url: &str) -> io::Result<Vec<u8>> {
    use std::time::Duration;
    let cfg = config();
    let mut b = ureq::Agent::config_builder()
        .user_agent(user_agent())
        .http_status_as_error(true)
        .max_redirects(10)
        .timeout_connect(Some(Duration::from_secs(cfg.timeout1.max(1))))
        .timeout_recv_response(Some(Duration::from_secs(cfg.timeout2.max(1))))
        .timeout_recv_body(Some(Duration::from_secs(cfg.timeout2.max(1))));
    if let Some(p) = proxy_url(&cfg) {
        let proxy = ureq::Proxy::new(&p).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("invalid proxy {}: {e}", cfg.proxy_host)))?;
        b = b.proxy(Some(proxy));
    }
    let agent: ureq::Agent = b.build().into();
    let mut resp = agent.get(url.trim()).call().map_err(|e| match e {
        ureq::Error::StatusCode(404) | ureq::Error::StatusCode(410) => io::Error::new(io::ErrorKind::NotFound, format!("{url}: not found (HTTP 404)")),
        ureq::Error::StatusCode(c) => io::Error::other(format!("{url}: server returned HTTP {c}")),
        ureq::Error::Timeout(_) => io::Error::new(io::ErrorKind::TimedOut, format!("{url}: connection timed out")),
        ureq::Error::HostNotFound => io::Error::new(io::ErrorKind::NotConnected, format!("{url}: host not found")),
        other => io::Error::other(format!("{url}: {other}")),
    })?;
    resp.body_mut().with_config().limit(2 << 30).read_to_vec().map_err(|e| io::Error::other(format!("{url}: {e}")))
}

/// Download `url`: unavailable in the browser build.
#[cfg(target_arch = "wasm32")]
pub fn fetch_url(url: &str) -> io::Result<Vec<u8>> {
    let _ = url;
    Err(io::Error::new(io::ErrorKind::Unsupported, NO_NETWORK))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxy_address() {
        let mut c = NetConfig::default();
        assert_eq!(proxy_url(&c), None);
        c.proxy_on = true;
        c.proxy_host = "proxy.example.org".into();
        c.proxy_port = 3128;
        assert_eq!(proxy_url(&c).unwrap(), "http://proxy.example.org:3128");
        c.proxy_auth = true;
        c.proxy_user = "me".into();
        c.proxy_pw = "p@ss".into();
        assert_eq!(proxy_url(&c).unwrap(), "http://me:p%40ss@proxy.example.org:3128");
        assert!(is_url("HTTPS://x.org/a") && !is_url("/tmp/a"));
    }
}
