
import psycopg2

try:
    connection = psycopg2.connect(
        host="localhost",
        port="5432",
        database="object_detection",
        user="postgres",
        password="3236"
    )

    print("PostgreSQL connection successful!")

    cursor = connection.cursor()
    cursor.execute("SELECT current_database(), current_user;")

    result = cursor.fetchone()
    print("Database:", result[0])
    print("User:", result[1])

    cursor.close()
    connection.close()

    print("Connection closed successfully.")

except Exception as error:
    print("Connection failed!")
    print(error)
