-- CustomerIQ: RFM Analysis SQL
-- Computes Recency, Frequency, Monetary for each customer
-- using completed orders only.
--
-- This query is run via DuckDB against the customeriq.duckdb database.

WITH completed_orders AS (
    SELECT
        o.order_id,
        o.user_id,
        o.created_at,
        o.status
    FROM orders o
    WHERE o.status IN ('complete', 'shipped', 'processing')
),

order_revenue AS (
    SELECT
        oi.order_id,
        oi.user_id,
        SUM(oi.sale_price) AS order_total
    FROM order_items oi
    WHERE oi.status IN ('complete', 'shipped', 'processing')
    GROUP BY oi.order_id, oi.user_id
),

observation AS (
    SELECT MAX(created_at) AS obs_date
    FROM completed_orders
),

customer_rfm AS (
    SELECT
        co.user_id                                        AS customer_id,
        MAX(co.created_at)                                AS last_purchase_date,
        DATE_DIFF('day', MAX(co.created_at), (SELECT obs_date FROM observation))
                                                          AS recency,
        COUNT(DISTINCT co.order_id)                       AS frequency,
        COALESCE(SUM(orv.order_total), 0)                 AS monetary
    FROM completed_orders co
    LEFT JOIN order_revenue orv
        ON co.order_id = orv.order_id
    GROUP BY co.user_id
)

SELECT
    customer_id,
    last_purchase_date,
    recency,
    frequency,
    ROUND(monetary, 2) AS monetary
FROM customer_rfm
ORDER BY monetary DESC;
