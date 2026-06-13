//! med-rag-assistant 边缘网关(M9)。
//!
//! 职责(典型的"前端 ↔ LLM 后端之间的轻量边缘层"):
//! - 托管 React 前端的静态构建产物
//! - POST /api/ask  → 反向代理到 Python FastAPI(MEDRAG_UPSTREAM)
//! - GET  /health   → 聚合网关自身 + 上游服务的健康状态
//! - 横切:请求日志(tracing)、上游超时
//!
//! 环境变量:
//!   GATEWAY_ADDR    监听地址,默认 0.0.0.0:3001
//!   MEDRAG_UPSTREAM Python 服务地址,默认 http://localhost:8000
//!   FRONTEND_DIST   前端构建目录,默认 ../frontend/dist

use std::time::Duration;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use tower_http::{services::ServeDir, trace::TraceLayer};

#[derive(Clone)]
struct AppState {
    client: reqwest::Client,
    upstream: String,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let addr = std::env::var("GATEWAY_ADDR").unwrap_or_else(|_| "0.0.0.0:3001".into());
    let upstream =
        std::env::var("MEDRAG_UPSTREAM").unwrap_or_else(|_| "http://localhost:8000".into());
    let dist = std::env::var("FRONTEND_DIST").unwrap_or_else(|_| "../frontend/dist".into());

    let state = AppState {
        client: reqwest::Client::builder()
            .timeout(Duration::from_secs(120)) // LLM 生成可能慢,放宽上游超时
            .build()
            .expect("build http client"),
        upstream,
    };

    let app = Router::new()
        .route("/api/ask", post(ask))
        .route("/health", get(health))
        .fallback_service(ServeDir::new(&dist))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    tracing::info!(%addr, "gateway listening");
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    axum::serve(listener, app).await.expect("serve");
}

/// 反向代理:把请求体原样转给上游 /ask,把上游响应(状态码 + JSON)原样带回。
async fn ask(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let url = format!("{}/ask", state.upstream);
    match state.client.post(&url).json(&body).send().await {
        Ok(resp) => {
            let status =
                StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let json = resp
                .json::<serde_json::Value>()
                .await
                .unwrap_or_else(|_| serde_json::json!({"detail": "upstream returned non-JSON"}));
            (status, Json(json))
        }
        Err(err) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"detail": format!("upstream unreachable: {err}")})),
        ),
    }
}

/// 健康聚合:网关自身 ok;上游打 /health,可达性一并上报。
async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let url = format!("{}/health", state.upstream);
    let upstream_ok = matches!(
        state.client.get(&url).timeout(Duration::from_secs(2)).send().await,
        Ok(resp) if resp.status().is_success()
    );
    let status = if upstream_ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(serde_json::json!({
            "gateway": "ok",
            "upstream": if upstream_ok { "ok" } else { "unreachable" },
        })),
    )
}
