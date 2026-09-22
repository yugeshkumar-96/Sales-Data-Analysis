import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("sales_data.csv")

# 2. Display basic information
print("First 5 records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:")
print(df.shape)

# 3. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Remove duplicate records
df = df.drop_duplicates()

# 5. Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# 6. Create Total Sales column
df["Total_Sales"] = df["Quantity"] * df["Unit_Price"]

# 7. Basic statistics
print("\nStatistical Summary:")
print(df.describe())

# 8. Total sales
total_sales = df["Total_Sales"].sum()
print("\nTotal Sales:", total_sales)

# 9. Sales by product
product_sales = df.groupby("Product")["Total_Sales"].sum()
print("\nSales by Product:")
print(product_sales.sort_values(ascending=False))

# 10. Sales by category
category_sales = df.groupby("Category")["Total_Sales"].sum()
print("\nSales by Category:")
print(category_sales.sort_values(ascending=False))

# 11. Monthly sales
df["Month"] = df["Date"].dt.to_period("M")

monthly_sales = df.groupby("Month")["Total_Sales"].sum()

print("\nMonthly Sales:")
print(monthly_sales)

# 12. Top 5 products
top_products = product_sales.sort_values(ascending=False).head(5)

print("\nTop 5 Products:")
print(top_products)

# 13. Visualization - Sales by Product
plt.figure(figsize=(10, 5))
product_sales.sort_values(ascending=False).plot(kind="bar")
plt.title("Sales by Product")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 14. Visualization - Sales by Category
plt.figure(figsize=(8, 5))
category_sales.plot(kind="bar")
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

# 15. Monthly sales trend
plt.figure(figsize=(10, 5))
monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# 16. Correlation heatmap
numeric_columns = df.select_dtypes(include="number")

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_columns.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
