# Scripted portfolio demo

Start the observable local stack with `make observable-stack`. Wait for
`http://localhost:8000/ready`, then keep `http://localhost:8000/metrics` open.
All identifiers and records used below are synthetic.

1. **Grounded response:** POST a high-confidence PIN-reset request to
   `/v1/conversations/respond`. Show the approved citation, provider metadata,
   response request ID, matching JSON log, metrics, and trace spans.
2. **Safe escalation:** repeat with confidence `0.42`, then request a money
   transfer. Show `low_confidence` and `sensitive_financial_request`; explain
   that escalation is policy behavior rather than model failure.
3. **Bilingual speech evidence:** show the governed benchmark taxonomy and the
   published ASR aggregate. If local ignored audio is present, run the frozen
   speech command; do not present synthetic voices as Filipino speakers.
4. **MCP boundary:** call the synthetic account tool unauthenticated, then with
   `accounts:read`. Call ticket creation without and then with explicit
   confirmation. Show denial/success audit outcomes without argument values.
5. **Interruption:** run the deterministic streaming test demonstrating that
   barge-in cancels an active response once and ignores stale completion.
6. **Evidence:** show `/metrics`, the sanitized trace example, the fault matrix,
   SLOs, load/cost table, and research report.

Close by stating the boundaries: deterministic local generation, synthetic
demo tools, no real bank access, no production users, no live credentials, and
no claim that the load result predicts external-model performance.
