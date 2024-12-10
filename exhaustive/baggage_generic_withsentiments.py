import pandas as pd
import random

# Templates for generic baggage inquiries
generic_prompts = [
    "What is the baggage weight limit for international flights?",
    "Can I carry extra baggage? If yes, how much will it cost?",
    "What is the maximum size of a cabin bag?",
    "Can you tell me about your airline's baggage policy?",
    "Is there any fee for overweight baggage?",
    "What are the items I cannot carry in my baggage?",
    "What should I do if my baggage is lost?",
    "How much baggage can I check in on domestic flights?",
    "What is the fee for extra baggage on international routes?",
    "Can I carry fragile items in my baggage?",
    "What is the process for reporting damaged baggage?",
    "Can you confirm the weight limit for carry-on baggage?",
    "How can I track my checked-in baggage?",
    "What is the process for claiming delayed baggage?",
    "Are there any restrictions on carrying liquids in hand luggage?",
    "Can I carry my pet as part of my baggage?",
    "What is the maximum number of bags allowed for international flights?",
    "Is there a fee for oversized baggage?",
    "Are musical instruments allowed as cabin baggage?",
    "Can I carry sports equipment as checked-in baggage?",
    "What should I do if my baggage is left behind at the airport?",
    "How do I request priority baggage handling?",
    "Can I purchase additional baggage allowance online?",
    "What happens if my baggage is delayed?",
    "Can you provide me with the baggage allowance details for economy class?",
    "Is there any compensation for lost baggage?",
    "Can I check in a stroller or car seat for free?",
    "Are there special guidelines for carrying electronic items in my baggage?",
    "What is the procedure for carrying medicines in checked-in baggage?",
    "Can I carry duty-free items as part of my hand luggage?",
    "What are the restrictions for carrying liquids and gels in cabin bags?",
    "Is there a charge for cabin baggage exceeding weight limits?",
    "What is the process for recovering lost baggage?",
    "Can I carry food items in my baggage?",
    "What is the maximum weight allowed for checked-in baggage?",
    "Are there any restrictions for carrying power banks in checked-in baggage?",
    "What is the procedure for carrying firearms or ammunition in my baggage?",
    "Can I carry sharp objects like scissors or knives in my checked luggage?",
    "Are there specific rules for carrying batteries in cabin baggage?",
    "Can you explain your airline's policy for damaged baggage claims?",
]

# Templates for flight-specific baggage inquiries
flight_specific_prompts = [
    "Where is my baggage from flight <MASKED_FLIGHT_NUM>?",
    "Has my baggage from flight <MASKED_FLIGHT_NUM> arrived?",
    "Can you confirm if my baggage is delayed?",
    "When can I expect my baggage to be delivered?",
    "Is there an issue with baggage from flight <MASKED_FLIGHT_NUM>?",
    "What is the status of baggage from flight <MASKED_FLIGHT_NUM>?",
    "I need an update on my baggage from flight <MASKED_FLIGHT_NUM>.",
    "Can you tell me if my baggage is still at the airport?",
    "Has my baggage been dispatched for delivery?",
    "Is my baggage delayed due to flight <MASKED_FLIGHT_NUM>?",
    "Can you check the status of my luggage?",
    "My baggage has not arrived. Can you help?",
    "I think my baggage is lost. What do I do now?",
    "Has baggage handling for flight <MASKED_FLIGHT_NUM> been completed?",
    "I need help finding my missing baggage.",
    "Did my baggage get transferred properly for my connecting flight?",
    "Is there a delay in baggage unloading for flight <MASKED_FLIGHT_NUM>?",
    "Why is my baggage not on the carousel for flight <MASKED_FLIGHT_NUM>?",
    "Can I retrieve my baggage from flight <MASKED_FLIGHT_NUM> early?",
    "What is the expected delivery time for baggage from flight <MASKED_FLIGHT_NUM>?",
    "Is baggage delivery from flight <MASKED_FLIGHT_NUM> affected by weather conditions?",
    "Can you confirm the terminal where baggage from flight <MASKED_FLIGHT_NUM> will be delivered?",
    "Did my baggage get misplaced during transit from flight <MASKED_FLIGHT_NUM>?",
    "Why is baggage unloading delayed for flight <MASKED_FLIGHT_NUM>?",
    "How do I file a complaint for baggage issues on flight <MASKED_FLIGHT_NUM>?",
    "Is the baggage carousel assigned for flight <MASKED_FLIGHT_NUM> functional?",
    "Where can I collect my baggage after flight <MASKED_FLIGHT_NUM>?",
    "Can you tell me the current location of my baggage from flight <MASKED_FLIGHT_NUM>?",
    "Was my baggage mishandled after flight <MASKED_FLIGHT_NUM>?",
    "How can I request an update on delayed baggage for flight <MASKED_FLIGHT_NUM>?",
    "What are the next steps for lost baggage from flight <MASKED_FLIGHT_NUM>?",
    "Was my baggage rechecked for my connecting flight?",
    "Has my baggage been sent to the wrong destination?",
    "How long will it take for my delayed baggage to arrive?",
    "What is the delay time for baggage from flight <MASKED_FLIGHT_NUM>?",
    "I haven’t received my baggage from flight <MASKED_FLIGHT_NUM>. What should I do?",
    "How do I track my delayed baggage from flight <MASKED_FLIGHT_NUM>?",
    "Can you tell me if my baggage is still at the departure airport?",
    "What is the status of the fragile item in my baggage?",
    "Can I update the delivery address for delayed baggage?",
]

# Responses
responses = {
    "ON_TIME": "Your baggage is on time and will be available shortly.",
    "DELAYED": "We regret the delay. Your baggage will be delivered soon.",
    "NO_STATUS": "We currently have no updates for your baggage. Please contact support.",
    "GENERIC": "Please check our baggage policy online or contact our team for details.",
}

# Sentiments
sentiments = ["positive", "neutral", "negative"]

# Generate dataset
data = []
statuses = ["ON_TIME", "DELAYED", "NO_STATUS", "GENERIC"]

for _ in range(5000):  # Generate 5000 generic and 5000 flight-specific prompts
    prompt = random.choice(generic_prompts + flight_specific_prompts)
    status = random.choice(statuses)
    response = responses[status]
    sentiment = random.choice(sentiments)
    data.append({"prompt": prompt, "status": status, "response": response, "sentiment": sentiment})

# Convert to DataFrame and save
df = pd.DataFrame(data)
df.to_csv("baggage_query_dataset.csv", index=False)
print("Dataset created and saved as baggage_query_dataset.csv")
