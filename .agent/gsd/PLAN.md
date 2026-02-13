# VEIL Stratified Implementation Plan

## Layer 1: Domain/Persistence
- [x] **Data Model**: Create `AuditLog` entity for Layer 7 ledger.
- [x] **Repository**: Implement `AuditLogRepository` (JPA/Postgres).
- [x] **E2E Test**: Verify Traffic Flow (Client -> Layer 0 -> Layer 3 -> Response).
- [x] **DTOs**: Define `VeilRequest`, `VeilResponse`, and `Verdict` enums.
- [x] **Test**: Unit test `AuditLogRepository` persistence.

## Layer 2: Business Logic (The Reflexes)
- [x] **Structure**: Define `SecurityLayer` interface for all 7 layers.
- [x] **L1 Identity**: Implement `RuntimeIdentityService` (PID/Hash verification).
- [x] **L2 Replay**: Implement `AntiReplayService` (Redis/JWT).
- [x] **L3 Rules**: Implement `DeterministicRuleEngine` (OPA/Java).
- [x] **L4 Council**: Implement `CouncilOfJudges` (Gemini SDK Stub/Integration).
- [x] **L5 Orchestrator**: Implement `TrafficLightService` to chain L1-L4.
- [x] **Test**: Unit test each Service independently (Mock dependencies).

## Layer 3: API Layer
- [x] **Interface**: Create `VeilCheckController` (REST Endpoint for Envoy ExtAuthz).
- [x] **Validation**: Add `@Valid` constraints to `VeilRequest` DTOs.
- [x] **Docs**: Generate OpenAPI spec for the auth interface.
- [x] **Test**: `MockMvc` tests for `VeilCheckController` inputs/outputs.

## Layer 0 & 4: Infrastructure & Integration (The Outer Shell)
- [x] **Layer 0 (Envoy)**: Create `envoy.yaml` with explicit `ext_authz` filter configuration.
- [x] **Docker**: Create `docker-compose.yaml` wiring Layer 0 (Envoy) to Layer 3 (API).
- [x] **Security**: Configure `SecurityConfig` to trust Layer 0 traffic.
- [x] **Test Report**: Generate comprehensive test report for L1-L3.
- [ ] **Layer 0 (Envoy)**: Create `envoy.yaml` with explicit `ext_authz` filter configuration.

## Phase 4: Advanced Hardening (Backlog)
- [ ] **Layer 6 (Response)**: Implement PII scanning on response bodies.
- [ ] **Layer 7 (Ledger)**: Implement Merkle Tree chaining for Audit Logs.
- [ ] **Layer 4 (Judges)**: Split Council into specific Context/Safety/Policy experts.
