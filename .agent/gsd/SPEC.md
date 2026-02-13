# VEIL - 7-Layer Sentinel Specification

## High-Level Goal
Build a "Living Identity Firewall" (Sidecar) for AI Agents that enforces a 7-layer security model on every outgoing packet. The system must be production-ready, using Java 17, Spring Boot 3.x, Envoy, and Gemini 3.0.

## Core Architecture (The Reflexes)
The system is an inescapable funnel. Traffic flows from Layer 0 -> Layer 7.

### Layer 0: The Smart Valve
- **Tech**: Envoy Proxy (Sidecar Pattern).
- **Function**: Intercepts all egress traffic from the Agent container via `iptables`/port forwarding (Port 10000).
- **Constraint**: Must be inescapable.

### Layer 1: Runtime Identity
- **Tech**: Java Service / SHA-256.
- **Function**: Verifies the Process ID (PID) and binary hash of the calling agent.
- **Goal**: Prevent supply chain attacks/code tampering.

### Layer 2: The Handshake (Anti-Replay)
- **Tech**: Redis + JWT.
- **Function**: Validates `Intent + Nonce`. Issues ephemeral 60s JWT.
- **Status**: *Foundation implementations exist, requires hardening.*

### Layer 3: Deterministic Policy
- **Tech**: Java Rule Engine / OPA.
- **Function**: Enforces hard binary rules (e.g., "No GET /admin", "No Spend > $500").
- **Goal**: >0ms rejection of obvious violations.

### Layer 4: The Council of Judges
- **Tech**: AI/LLM (Gemini Flash/Nano).
- **Function**: Mixture of Experts (Context, Safety, Policy) voting on fuzzy risks (e.g., Prompt Injection).
- **Goal**: Deep semantic analysis of intent.

### Layer 5: Traffic Light Service (Controller)
- **Tech**: Java Logic.
- **Function**: The central "Brain of the Reflex". Orchestrates the flow:
  - **Green Lane**: Fast path (Read-only).
  - **Yellow Lane**: Audit path (Low risk).
  - **Red Lane**: Blocking path (High risk -> Council).

### Layer 6: Execution Gate
- **Tech**: Envoy / Java HttpClient.
- **Function**: Proxies the actual request to the external world.
- **Feature**: Scans *responses* for PII leakage before returning to Agent.

### Layer 7: Verifiable Ledger
- **Tech**: Merkle Tree / PostgreSQL.
- **Function**: Cryptographically logs every decision (Allow/Block) for immutable forensics.

## Technical Constraints
- **Language**: Java 17 (Spring Boot 3.x)
- **Testing**: JUnit 5, TestContainers (Strict TDD)
- **AI**: Google Gemini SDK
- **Data**: Redis (Hot), PostgreSQL (Cold)

## User Stories (High Level)
1. As an Agent, I cannot bypass the Envoy proxy.
2. As a Security Officer, I want guaranteed logging of every network call (Layer 7).
3. As the System, I must block specific URL paths (Layer 3) instantly.
4. As the System, I must detect prompt attacks via AI analysis (Layer 4).
