-- CustomerIQ: Category Analytics SQL
-- Revenue, volume, and customer count by product category.

SELECT
    p.category,
    ROUND(SUM(oi.sale_price), 2)       AS total_revenue,
    COUNT(DISTINCT oi.order_id)        AS order_count,
    COUNT(DISTINCT oi.user_id)         AS unique_customers,
    ROUND(AVG(oi.sale_price), 2)       AS avg_selling_price,
    COUNT(DISTINCT oi.product_id)      AS product_count
FROM order_items oi
JOIN products p ON oi.product_id = p.id
WHERE oi.status IN ('complete', 'shipped', 'processing')
  AND p.category IS NOT NULL
GROUP BY p.category
ORDER BY total_revenue DESC;
