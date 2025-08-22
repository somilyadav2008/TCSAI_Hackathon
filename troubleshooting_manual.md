Of course. Here is the content for the troubleshooting_manual.md file based on the information you provided.
You can copy and paste this directly into your troubleshooting_manual.md file.
# 🏦 Banking Management System – Troubleshooting Manual

---

### 🔐 Login Service

* **Common Issues:**
    * Session timeout misconfiguration
    * Multiple failed logins → account lockout
* **Resolution Steps:**
    1.  Check session policy in `auth_config.yaml`.
    2.  Ensure session DB (Redis/Memcache) is not evicting entries.
    3.  For account lockout → verify lockout threshold; manually unlock in `user_mgmt` table if false positive.

---

### 💳 Transaction Engine

* **Issue: Transaction Deadlock / Rollback**
    * **Cause:** Conflicting queries on same rows.
    * **Resolution:**
        1.  Identify blocking queries (`SHOW ENGINE INNODB STATUS`).
        2.  Optimize stored procedures for order of operations.
        3.  Implement retry mechanism with exponential backoff.

* **Issue: Duplicate Transaction**
    * **Cause:** Double-clicking, retry without unique txn ID.
    * **Resolution:**
        1.  Enforce unique transaction ID at middleware.
        2.  Add DB unique constraint on `transaction_id`.
        3.  Check reconciliation script for duplicate cleanup.

---

### 🏦 Payment Gateway

* **Issue: Timeout with External API**
    * **Cause:** Network latency, firewall, DNS issue.
    * **Resolution:**
        1.  Run `ping`/`traceroute` to payment endpoint.
        2.  Verify firewall outbound port 443 rules.
        3.  Switch to backup gateway if downtime > 30s.

* **Issue: Invalid Checksum**
    * **Cause:** Mismatched encryption keys, payload tampering.
    * **Resolution:**
        1.  Validate key rotation logs.
        2.  Compare checksum algorithm versions.
        3.  Re-sync secret keys with payment provider.

---

### 📂 Loan Processing

* **Issue: Batch Job Stuck**
    * **Cause:** Large dataset, memory leak.
    * **Resolution:**
        1.  Review job scheduler logs.
        2.  Increase JVM heap size.
        3.  Break batch into smaller chunks.

* **Issue: EMI Auto-Debit Failure**
    * **Cause:** Mandate expired, insufficient funds.
    * **Resolution:**
        1.  Check ECS/NACH mandate expiry.
        2.  Retry after 24h for insufficient balance.
        3.  Generate customer notification.

---

### 🗄 Database Layer

* **High CPU Usage**
    * **Cause:** Unoptimized queries, missing indexes.
    * **Resolution:**
        1.  Run slow query log.
        2.  Create missing indexes.
        3.  Shift reporting jobs off-peak.

* **Backup Job Failure**
    * **Cause:** Disk full.
    * **Resolution:**
        1.  Clear old logs/backups.
        2.  Add storage or move backups to cloud.
        3.  Automate retention policy.

---

### 💰 Core Banking

* **Account Balance Mismatch**
    * **Cause:** Transaction not committed, duplicate.
    * **Resolution:**
        1.  Compare transaction logs with reconciliation tables.
        2.  Trigger re-run of reconciliation job.
        3.  Escalate to Finance if mismatch persists.

* **Interest Calculation Error**
    * **Cause:** Wrong formula update.
    * **Resolution:**
        1.  Verify calculation scripts in `interest_calc.py`.
        2.  Patch formula and rerun test dataset.
        3.  Release hotfix.

---

### 📊 Reporting Module

* **End-of-Day Report Failure**
    * **Cause:** Missing data due to incomplete jobs.
    * **Resolution:**
        1.  Identify failed jobs from job monitor.
        2.  Re-run ETL pipeline.
        3.  Generate report manually if urgent.

* **Incorrect Report Figures**
    * **Cause:** Formula bug, stale cache.
    * **Resolution:**
        1.  Clear reporting cache.
        2.  Verify SQL queries in reporting layer.
        3.  Cross-check against DB snapshots.

---

### 📢 Notification Service

* **SMS Gateway Unreachable**
    * **Cause:** Vendor downtime.
    * **Resolution:**
        1.  Switch to backup SMS vendor.
        2.  Queue pending messages.
        3.  Retry in batches.

* **Email Queue Backlog**
    * **Cause:** Mail server rate limits.
    * **Resolution:**
        1.  Check mail queue (`postqueue -p`).
        2.  Add worker threads.
        3.  Throttle email sending.

---

### 🔒 Security & Audit

* **Suspicious Admin Login**
    * **Cause:** Credential compromise.
    * **Resolution:**
        1.  Block suspicious IP immediately.
        2.  Force password reset.
        3.  Trigger security audit.

