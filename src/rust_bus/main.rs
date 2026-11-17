//! ─────────────────────────────────────────────────────────────────────────────
//! echo-agent · echo-bus: Rust actor-model IPC event bus
//!
//! Speaks newline-delimited JSON-RPC 2.0 over a local Unix domain socket
//! (or TCP on 127.0.0.1 as a fallback). This is the high-throughput backbone
//! used by internal sub-agents (Coder / Researcher / Debugger) and bridges to
//! the Python microkernel's Pub/Sub with a thin adapter.
//!
//! Wire protocol (one JSON document per line):
//!   → {"jsonrpc":"2.0","id":1,"method":"publish","params":{"topic":"agent:coder","payload":{...}}}
//!   ← {"jsonrpc":"2.0","id":1,"result":{"delivered":3}}
//!   → {"jsonrpc":"2.0","id":2,"method":"subscribe","params":{"topic":"agent:*"}}
//!   ← {"jsonrpc":"2.0","id":2,"result":{"subscribed":true}}
//!   ← {"jsonrpc":"2.0","method":"event","params":{"topic":"...","payload":...}}   (push)
//!   → {"jsonrpc":"2.0","id":3,"method":"ping"}     ← {"result":"pong"}
//!   → {"jsonrpc":"2.0","id":4,"method":"stats"}    ← {"result":{...}}
//!
//! Design notes:
//!   * Actor model: a single `Broker` task owns all subscription state — the
//!     only place mutation happens. No locks on the hot path, no data races.
//!   * Connections never touch shared state; they talk to the Broker through
//!     an mpsc command channel.
//!   * Topics support trailing-glob matching ("agent:*") and the catch-all "*".
//! ─────────────────────────────────────────────────────────────────────────────

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, UnixListener};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Unique identifier for each connected client.
type ConnId = u64;

/// Commands sent from connection tasks to the central Broker actor.
#[derive(Debug)]
enum BrokerCommand {
    /// Register a subscription. `tx` is the connection's outbound writer,
    /// which the Broker also clones for event fan-out.
    Subscribe {
        conn: ConnId,
        topic: String,
        tx: mpsc::UnboundedSender<String>,
    },
    /// Drop all subscriptions belonging to a connection (on disconnect).
    Unsubscribe { conn: ConnId },
    /// Publish a payload to a topic; reports how many subscribers received it.
    Publish {
        topic: String,
        payload: Value,
        reply: mpsc::Sender<usize>,
    },
    /// Request a snapshot of bus statistics.
    Stats { reply: mpsc::Sender<Value> },
}

/// The single point of truth for routing tables.
#[derive(Default)]
struct Broker {
    /// conn -> topics it subscribed to (for O(1) disconnect cleanup).
    by_conn: HashMap<ConnId, Vec<String>>,
    /// topic filter -> subscriber connections.
    by_topic: HashMap<String, HashSet<ConnId>>,
    /// conn -> outbound frame channel (Publish fans out without any locks).
    writers: HashMap<ConnId, mpsc::UnboundedSender<String>>,
}

/// Glob semantics: "*" matches everything; "agent:*" is a prefix match.
fn topic_matches(topic: &str, filter: &str) -> bool {
    if filter == "*" {
        true
    } else if let Some(prefix) = filter.strip_suffix('*') {
        topic.starts_with(prefix)
    } else {
        topic == filter
    }
}

impl Broker {
    fn subscribe(&mut self, conn: ConnId, topic: String, tx: mpsc::UnboundedSender<String>) {
        self.writers.insert(conn, tx);
        self.by_topic.entry(topic.clone()).or_default().insert(conn);
        self.by_conn.entry(conn).or_default().push(topic);
        debug!(conn, "subscribed");
    }

    fn unsubscribe(&mut self, conn: ConnId) {
        if let Some(topics) = self.by_conn.remove(&conn) {
            for t in topics {
                if let Some(set) = self.by_topic.get_mut(&t) {
                    set.remove(&conn);
                    if set.is_empty() {
                        self.by_topic.remove(&t);
                    }
                }
            }
        }
        self.writers.remove(&conn);
    }

    /// Fan a payload out to every connection whose filters match the topic.
    async fn publish(&mut self, topic: &str, payload: &Value) -> usize {
        let mut targets: HashSet<ConnId> = HashSet::new();
        for (filter, conns) in &self.by_topic {
            if topic_matches(topic, filter) {
                targets.extend(conns.iter().copied());
            }
        }
        if targets.is_empty() {
            return 0;
        }
        let frame = json!({
            "jsonrpc": "2.0",
            "method": "event",
            "params": { "topic": topic, "payload": payload },
        })
        .to_string();
        let mut delivered = 0usize;
        for conn in targets {
            if let Some(tx) = self.writers.get(&conn) {
                // A closed writer means the peer died; its read loop will
                // notice shortly and send Unsubscribe — safe to skip.
                if tx.send(frame.clone()).is_ok() {
                    delivered += 1;
                }
            }
        }
        delivered
    }

    fn stats(&self) -> Value {
        json!({
            "connections": self.writers.len(),
            "subscriptions": self.by_conn.values().map(Vec::len).sum::<usize>(),
            "topics": self.by_topic.len(),
        })
    }
}

/// The Broker actor: the only task that ever mutates routing state.
async fn broker_actor(mut rx: mpsc::Receiver<BrokerCommand>) {
    let mut broker = Broker::default();
    while let Some(cmd) = rx.recv().await {
        match cmd {
            BrokerCommand::Subscribe { conn, topic, tx } => broker.subscribe(conn, topic, tx),
            BrokerCommand::Unsubscribe { conn } => broker.unsubscribe(conn),
            BrokerCommand::Publish { topic, payload, reply } => {
                let delivered = broker.publish(&topic, &payload).await;
                let _ = reply.send(delivered).await;
            }
            BrokerCommand::Stats { reply } => {
                let _ = reply.send(broker.stats()).await;
            }
        }
    }
    info!("broker actor shutting down");
}

/// One parsed inbound JSON-RPC request.
#[derive(Deserialize)]
struct RpcRequest {
    method: String,
    #[serde(default)]
    id: Option<Value>,
    #[serde(default)]
    params: Value,
}

/// Serialize a success response; None for notifications (no `id` present).
fn respond(id: &Option<Value>, result: Value) -> Option<String> {
    id.as_ref()
        .map(|id| json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string())
}

fn respond_error(id: &Option<Value>, code: i64, message: &str) -> Option<String> {
    id.as_ref().map(|id| {
        json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
            .to_string()
    })
}

/// Per-connection handler: parse inbound frames, dispatch to the Broker, and
/// pump outbound frames (RPC responses + broker event pushes) to the socket.
async fn handle_conn<R, W>(conn_id: ConnId, read_half: R, write_half: W, cmd_tx: mpsc::Sender<BrokerCommand>)
where
    R: AsyncRead + Unpin,
    // Send + 'static: the writer half is moved into a dedicated spawned pump
    // task below, so the compiler must be able to prove it can cross tasks.
    W: AsyncWrite + Unpin + Send + 'static,
{
    // Outbound queue shared by this handler (RPC responses) and the Broker
    // (event fan-out). Single writer task = no interleaved partial frames.
    let (frame_tx, mut frame_rx) = mpsc::unbounded_channel::<String>();
    let pump = tokio::spawn(async move {
        let mut out = write_half;
        let mut buffer = String::new();
        while let Some(frame) = frame_rx.recv().await {
            buffer.push_str(&frame);
            buffer.push('\n');
            // Micro-batch: flush when the buffer is large or the queue drains,
            // minimizing syscalls under load without hurting latency.
            if buffer.len() >= 16 * 1024 || frame_rx.is_empty() {
                if out.write_all(buffer.as_bytes()).await.is_err() {
                    break;
                }
                let _ = out.flush().await;
                buffer.clear();
            }
        }
    });

    let mut reader = BufReader::new(read_half).lines();
    while let Ok(Some(line)) = reader.next_line().await {
        if line.trim().is_empty() {
            continue;
        }
        let req: RpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                if let Some(err) = respond_error(&None, -32700, &format!("parse error: {e}")) {
                    let _ = frame_tx.send(err);
                }
                continue;
            }
        };
        let id = req.id.clone();
        match req.method.as_str() {
            "publish" => {
                let topic = req.params["topic"].as_str().unwrap_or_default().to_string();
                if topic.is_empty() {
                    if let Some(err) = respond_error(&id, -32602, "params.topic is required") {
                        let _ = frame_tx.send(err);
                    }
                    continue;
                }
                let (reply_tx, mut reply_rx) = mpsc::channel::<usize>(1);
                let _ = cmd_tx
                    .send(BrokerCommand::Publish {
                        topic,
                        payload: req.params["payload"].clone(),
                        reply: reply_tx,
                    })
                    .await;
                let delivered = reply_rx.recv().await.unwrap_or(0);
                if let Some(resp) = respond(&id, json!({ "delivered": delivered })) {
                    let _ = frame_tx.send(resp);
                }
            }
            "subscribe" => {
                let topic = req.params["topic"].as_str().unwrap_or("*").to_string();
                // Hand the Broker a clone of frame_tx: subsequent matching
                // events are pushed on this same connection.
                let _ = cmd_tx
                    .send(BrokerCommand::Subscribe {
                        conn: conn_id,
                        topic,
                        tx: frame_tx.clone(),
                    })
                    .await;
                if let Some(resp) = respond(&id, json!({ "subscribed": true })) {
                    let _ = frame_tx.send(resp);
                }
            }
            "ping" => {
                if let Some(resp) = respond(&id, json!("pong")) {
                    let _ = frame_tx.send(resp);
                }
            }
            "stats" => {
                let (reply_tx, mut reply_rx) = mpsc::channel::<Value>(1);
                let _ = cmd_tx.send(BrokerCommand::Stats { reply: reply_tx }).await;
                let stats = reply_rx.recv().await.unwrap_or_else(|| json!({}));
                if let Some(resp) = respond(&id, stats) {
                    let _ = frame_tx.send(resp);
                }
            }
            other => {
                if let Some(err) = respond_error(&id, -32601, &format!("unknown method: {other}")) {
                    let _ = frame_tx.send(err);
                }
            }
        }
    }

    // Peer disconnected: drop its subscriptions and let the pump finish.
    let _ = cmd_tx.send(BrokerCommand::Unsubscribe { conn: conn_id }).await;
    debug!(conn = conn_id, "connection closed");
    let _ = pump.abort();
}

/// Either listener kind, unified behind one async accept path. Tokio's
/// `accept()` is a `Future`, so the old sync `FnMut` closure shape cannot
/// work without blocking the runtime; an enum keeps the two variants honest.
enum Listener {
    Unix(UnixListener),
    Tcp(TcpListener),
}

impl Listener {
    /// Accept one client and split it into owned (read, write) halves.
    async fn accept(
        &self,
    ) -> std::io::Result<(
        Box<dyn AsyncRead + Unpin + Send>,
        Box<dyn AsyncWrite + Unpin + Send>,
    )> {
        match self {
            Listener::Unix(l) => {
                let (stream, _) = l.accept().await?;
                let (r, w) = stream.into_split();
                Ok((
                    Box::new(r) as Box<dyn AsyncRead + Unpin + Send>,
                    Box::new(w) as Box<dyn AsyncWrite + Unpin + Send>,
                ))
            }
            Listener::Tcp(l) => {
                let (stream, _) = l.accept().await?;
                let (r, w) = stream.into_split();
                Ok((
                    Box::new(r) as Box<dyn AsyncRead + Unpin + Send>,
                    Box::new(w) as Box<dyn AsyncWrite + Unpin + Send>,
                ))
            }
        }
    }
}

/// Accept loop shared by Unix and TCP listeners.
async fn accept_loop(
    cmd_tx: mpsc::Sender<BrokerCommand>,
    listener: Listener,
) -> std::io::Result<()> {
    let mut next_conn = 1u64;
    loop {
        // Accept a client; transient errors are logged and retried.
        let (read_half, write_half) = match listener.accept().await {
            Ok(halves) => halves,
            Err(e) => {
                warn!(error = %e, "accept failed; continuing");
                continue;
            }
        };
        let conn_id = next_conn;
        next_conn += 1;
        let cmd = cmd_tx.clone();
        tokio::spawn(handle_conn(conn_id, read_half, write_half, cmd));
        debug!(conn = conn_id, "client connected");
        // Yield briefly so the runtime can schedule the freshly spawned task.
        tokio::task::yield_now().await;
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "echo_bus=info".into()),
        )
        .init();

    // Minimal arg parsing: --socket PATH | --tcp PORT (default: Unix socket).
    let mut args = std::env::args().skip(1);
    let mut socket_path: Option<String> = None;
    let mut tcp_port: Option<u16> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--socket" => socket_path = args.next(),
            "--tcp" => tcp_port = args.next().and_then(|p| p.parse().ok()),
            _ => {}
        }
    }

    let (cmd_tx, cmd_rx) = mpsc::channel::<BrokerCommand>(4096);
    tokio::spawn(broker_actor(cmd_rx));
    info!("echo-bus broker actor online");

    let socket = socket_path.or_else(|| std::env::var("ECHO_BUS_SOCKET").ok());

    if let Some(path) = socket {
        // Unix domain socket (preferred: zero network stack overhead).
        let _ = std::fs::remove_file(&path); // clear stale socket after a crash
        let listener = UnixListener::bind(&path)?;
        info!(socket = %path, "echo-bus listening (unix)");
        accept_loop(cmd_tx, Listener::Unix(listener)).await?;
        let _ = std::fs::remove_file(&path);
    } else {
        // TCP fallback for environments without Unix sockets (rare locally).
        let port = tcp_port.unwrap_or(7531);
        let listener = TcpListener::bind(("127.0.0.1", port)).await?;
        info!(port, "echo-bus listening (tcp/127.0.0.1)");
        accept_loop(cmd_tx, Listener::Tcp(listener)).await?;
    }
    Ok(())
}

// `Serialize` is required for the JSON payload types; silence unused warning
// if the compiler determines it's only used transitively.
#[allow(dead_code)]
fn _assert_serde() {
    fn assert<T: Serialize>() {}
    assert::<Value>();
}
