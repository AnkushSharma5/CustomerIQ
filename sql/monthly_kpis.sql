-- CustomerIQ: Monthly KPI Analytics SQL
-- Calculates key business metrics per calendar month.

WITH completed_orders AS (
    SELECT
        o.order_id,
        o.user_id,
        DATE_TRUNC('month', o.created_at) AS order_month,
        o.created_at
    FROM orders o
    WHERE o.status IN ('complete', 'shipped', 'processing')
),

order_revenue AS (
    SELECT
        oi.order_id,
        SUM(oi.sale_price) AS order_total
    FROM order_items oi
    WHERE oi.status IN ('complete', 'shipped', 'processing')
    GROUP BY oi.order_id
),

monthly_base AS (
    SELECT
        co.order_month,
        co.order_id,
        co.user_id,
        COALESCE(orv.order_total, 0) AS order_total
    FROM completed_orders co
    LEFT JOIN order_revenue orv ON co.order_id = orv.order_id
),

first_orders AS (
    SELECT user_id, MIN(order_month) AS first_month
    FROM monthly_base
    GROUP BY user_id
)

SELECT
    STRFTIME(mb.order_month, '%Y-%m')                           AS year_month,
    ROUND(SUM(mb.order_total), 2)                               AS total_revenue,
    COUNT(DISTINCT mb.order_id)                                 AS total_orders,
    COUNT(DISTINCT mb.user_id)                                  AS unique_customers,
    ROUND(SUM(mb.order_total) / COUNT(DISTINCT mb.order_id), 2) AS avg_order_value,
    COUNT(DISTINCT CASE WHEN mb.order_month = fo.first_month THEN mb.user_id END)
                                                                AS new_customers,
    COUNT(DISTINCT CASE WHEN mb.order_month != fo.first_month THEN mb.user_id END)
                                                                AS repeat_customers
FROM monthly_base mb
LEFT JOIN first_orders fo ON mb.user_id = fo.user_id
GROUP BY mb.order_month
ORDER BY mb.order_month;
