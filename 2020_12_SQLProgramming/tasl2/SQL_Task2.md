

# Task 2  [查询与排序](http://datawhale.club/t/topic/476)

[组队学习](http://datawhale.club/c/team-learning/5)[SQL编程语言]
## SQL Exercise
### 2.1

    SELECT
    product_name, regist_date
    FROM
    product
    WHERE
    regist > "2009/04/28";
   
   ### 2.2
   #### 1
   

    SELECT * FROM product 
    WHERE purchase_price = NULL;


#### 2

    SELECT * FROM product 
    WHERE purchase_price <> NULL;


#### 3

    SELECT * FROM product 
    WHERE product_name > NULL;
    
   返回值为空，条件语句应为product_name IS NULL 或者 product_name IS NOT NULL
### 2.3


 

    SELECT product_name, sale_price, purchase_price
    FROM product
    WHERE 
    (sale_price - purchase_price) >= 500;

    SELECT product_name, sale_price, purchase_price
    FROM product
    WHERE 
    (purchase_priec + 500) <= sale_price;

### 2.4

    SELECT 
    product_name, product_type, (sale_price * 0.9 - purchase_price) AS "profit"
    FROM product
    WHERE 
    (sale_price * 0.9 - purchase_price) >= 100;

### 2.5

请指出下述SELECT语句中所有的语法错误。

```
SELECT product_id, SUM（product_name）
--本SELECT语句中存在错误。
  FROM product 
 GROUP BY product_type 
 WHERE regist_date > '2009-09-01';
```
   作为GROUP BY 的特征product_type却没有被选择

### 2.6

    SELECT 
    product_type, SUM(sale_price), SUM(purchase_price)
    FROM product
    GROUP BY product_type
    WHERE 
    SUM(sale_price) > (SUM(purchase_price) * 1.5)

### 2.7

    SELECT * 
    FROM product 
    ORDER BY 
    (sale_price / purchase_price)


