import csv

# Define the dataset with intents and corresponding user queries
intent_data = {
    "flight_status": [
        "What is the status of flight 123?",
        "Can you tell me if flight 456 is on time?",
        "Is flight 789 delayed?",
        "Can I get an update on flight ABC?",
        "Has flight XYZ already departed?"
    ],
    "book_flight": [
        "I want to book a flight.",
        "Can you help me book a ticket?",
        "I need to book a flight to New York.",
        "How do I reserve a flight?",
        "Book a flight for me, please."
    ],
    "cancel_ticket": [
        "Cancel my ticket.",
        "I need to cancel my flight reservation.",
        "Can you help me cancel my booking?",
        "Cancel the ticket for flight 123.",
        "How do I cancel my ticket?"
    ],
    "reschedule_flight": [
        "I need to reschedule my flight.",
        "Can you help me change my flight time?",
        "Reschedule my flight to tomorrow.",
        "How do I postpone my flight?",
        "Can I reschedule my booking for next week?"
    ],
    "baggage_policy": [
        "What is your baggage policy?",
        "How much luggage can I carry?",
        "Tell me about the baggage allowance.",
        "Is there a fee for extra luggage?",
        "What are the rules for hand baggage?"
    ]
}

# Define the output CSV file
output_csv = "intent_dataset.csv"

# Write data to CSV
with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["text", "intent"])
    # Write each intent and its associated queries
    for intent, queries in intent_data.items():
        for query in queries:
            writer.writerow([query, intent])

print(f"Dataset successfully created and saved to {output_csv}")
