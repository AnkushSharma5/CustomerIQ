-- CustomerIQ: Cohort Retention SQL
-- Computes monthly cohort retention rates.

WITH completed_orders AS (
    SELECT
        o.order_id,
        o.user_id,
        DATE_TRUNC('month', o.created_at) AS order_month
    FROM orders o
    WHERE o.status IN ('complete', 'shipped', 'processing')
),

cohort_assignments AS (
    SELECT
        user_id,
        MIN(order_month) AS cohort_month
    FROM completed_orders
    GROUP BY user_id
),

cohort_sizes AS (
    SELECT cohort_month, COUNT(DISTINCT user_id) AS cohort_size
    FROM cohort_assignments
    GROUP BY cohort_month
),

monthly_activity AS (
    SELECT
        co.user_id,
        co.order_month,
        ca.cohort_month,
        DATEDIFF('month', ca.cohort_month, co.order_month) AS months_since
    FROM completed_orders co
    JOIN cohort_assignments ca ON co.user_id = ca.user_id
),

cohort_activity AS (
    SELECT
        cohort_month,
        months_since,
        COUNT(DISTINCT user_id) AS active_customers
    FROM monthly_activity
    GROUP BY cohort_month, months_since
)

SELECT
    STRFTIME(ca.cohort_month, '%Y-%m') AS cohort_month,
    ca.months_since,
    ca.active_customers,
    cs.cohort_size,
    ROUND(ca.active_customers * 1.0 / cs.cohort_size, 4) AS retention_rate
FROM cohort_activity ca
JOIN cohort_sizes cs ON ca.cohort_month = cs.cohort_month
ORDER BY ca.cohort_month, ca.months_since;
