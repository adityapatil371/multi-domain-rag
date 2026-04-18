from google.cloud import bigquery
import pandas as pd

def analyse_query_logs():
    client = bigquery.Client(project="ml-portfolio-493708")
    
    # Query 1 — queries per domain
    query1 = """
        SELECT domain, COUNT(*) as query_count
        FROM `ml-portfolio-493708.rag_analytics.query_logs`
        GROUP BY domain
        ORDER BY query_count DESC
    """
    
    # Query 2 — confidence rate per domain
    query2 = """
        SELECT 
            domain,
            COUNTIF(confident = TRUE) as confident_count,
            COUNT(*) as total,
            ROUND(COUNTIF(confident = TRUE) / COUNT(*) * 100, 1) as confidence_rate_pct
        FROM `ml-portfolio-493708.rag_analytics.query_logs`
        GROUP BY domain
    """
    
    # Query 3 — most recent questions
    query3 = """
        SELECT timestamp, domain, question
        FROM `ml-portfolio-493708.rag_analytics.query_logs`
        ORDER BY timestamp DESC
        LIMIT 5
    """
    
    print("=== Queries per Domain ===")
    df1 = client.query(query1).to_dataframe()
    print(df1.to_string(index=False))
    
    print("\n=== Confidence Rate per Domain ===")
    df2 = client.query(query2).to_dataframe()
    print(df2.to_string(index=False))
    
    print("\n=== Most Recent Questions ===")
    df3 = client.query(query3).to_dataframe()
    print(df3.to_string(index=False))

if __name__ == "__main__":
    analyse_query_logs()