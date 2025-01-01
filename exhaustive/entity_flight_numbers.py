import csv
import random

# Airlines and flight formats
airlines = ["AA", "BA", "DL", "EK", "QR", "AF", "CX", "LH", "UL", "QF"]
flight_numbers = [str(random.randint(100, 9999)) for _ in range(200)]

# Sentences
templates = [
    "What is the status of my flight {}?",
    "Has {} landed yet?",
    "My flight number is {}.",
    "When will {} depart?",
    "Please let me know about flight {}.",
    "I missed flight {} this morning.",
    "Flight {} was delayed due to weather.",
    "Is there any update for flight {}?",
    "Flight {} has been rescheduled.",
    "Your booking for flight {} is confirmed."
]

# Generate 200 samples
samples = []
for i in range(200):
    airline = random.choice(airlines)
    flight_number = airline + random.choice(flight_numbers)
    sentence = random.choice(templates).format(flight_number)
    start = sentence.find(flight_number)
    end = start + len(flight_number)
    entity = [{"start": start, "end": end, "label": "B-FLIGHT"}]
    samples.append({"text": sentence, "entities": entity})

# Save to CSV
csv_file = "flight_number_entity.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "entities"])
    writer.writeheader()
    for sample in samples:
        writer.writerow({"text": sample["text"], "entities": sample["entities"]})

print(f"CSV file with 200 samples saved as {csv_file}.")
