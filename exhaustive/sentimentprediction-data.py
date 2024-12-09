import csv
import random

# Define airline-specific sentiment samples
positive_samples = [
        "This is the best thing ever!",
        "I’m so happy with this purchase.",
        "Fantastic quality and service.",
        "Absolutely love this product!",
        "I couldn’t ask for more.",
        "Perfect in every way!",
        "This is worth every penny.",
        "Amazing! Highly recommend.",
        "Exceptional value for the money.",
        "I’m thrilled with how it turned out.",
        "Everything about this is great.",
        "I couldn’t be more satisfied.",
        "Wonderful product, thank you!",
        "This has exceeded my expectations.",
        "It’s a joy to use this every day.",
        "I feel so lucky to have found this.",
        "This makes my life so much easier.",
        "Highly recommend to all my friends.",
        "Outstanding! Can’t imagine life without it.",
        "Such a pleasant surprise!",
        "Absolutely delighted with this!",
        "The quality is unmatched.",
        "I’m beyond impressed!",
        "Flawless from start to finish.",
        "So glad I decided to try this.",
        "This makes me so happy!",
        "Top-notch product!",
        "I would buy this again in a heartbeat.",
        "This has made such a difference in my life.",
        "I love everything about this!",
        "Thank you for resolving my flight issue so quickly!",
        "The customer service was fantastic. I appreciate the help!",
        "I’m so pleased with how smoothly everything went.",
        "Amazing experience with your airline today!",
        "The flight arrived on time, and everything was perfect.",
        "I appreciate the prompt updates about my PNR <MASKED_PNR>.",
        "Booking with you was so easy and efficient!",
        "The crew was very polite and helpful. Thank you!",
        "I’m so happy my luggage arrived safely and on time.",
        "This is the best airline service I’ve ever experienced.",
        "I loved how you handled my rebooking request for <MASKED_FLIGHT_NUM>.",
        "Thank you for the timely status update about my flight.",
        "Everything was great, from check-in to boarding.",
        "I’m so satisfied with the level of care your staff provided.",
        "I’ll definitely be flying with you again!",
        "Your mobile app made the PNR lookup process so simple.",
        "Fantastic support in helping me change my seat!",
        "Thanks for making my travel experience stress-free.",
        "I appreciate the proactive delay notification for flight <MASKED_FLIGHT_NUM>.",
        "The boarding process was seamless and well-organized.",
        "The flight was on time and the service was fantastic!",
        "I loved how the staff handled everything so efficiently.",
        "The boarding process was smooth and hassle-free. Great job!",
        "Thank you for the complimentary snacks during the flight!",
        "The seats were so comfortable. I’m impressed by the service!",
        "Amazing experience with Ace Airlines. Keep it up!",
        "The in-flight entertainment was superb. Loved it!",
        "Thank you for making my journey enjoyable and stress-free.",
        "I appreciated the warm welcome by the flight attendants.",
        "The customer support team resolved my issue quickly. Excellent!"
    ]
neutral_samples = [
        "The flight was average, nothing special.",
        "The service was okay but could be better.",
        "I reached my destination without any issues.",
        "The staff was neither friendly nor rude.",
        "The food was acceptable, not the best I’ve had.",
        "The overall experience was decent, no major complaints.",
        "The flight landed on time, but the takeoff was delayed.",
        "I got my luggage without any hassle. Just a typical day.",
        "The cabin temperature was fine, not too cold or hot.",
        "It was a regular flight. Nothing stood out."
        "It’s fine, I guess.",
        "Does what it says on the tin.",
        "Nothing special, just okay.",
        "Meets the basic requirements.",
        "It’s adequate for now.",
        "Serves its purpose.",
        "Not bad, not great.",
        "It’s alright, I suppose.",
        "A standard, functional product.",
        "Nothing to write home about.",
        "Gets the job done.",
        "It’s as expected.",
        "I have no strong opinions about this.",
        "It’s a very average experience.",
        "It does what it’s supposed to do.",
        "Neither good nor bad.",
        "It’s acceptable.",
        "The quality is what I expected.",
        "I feel indifferent about this.",
        "No complaints, but no praises either.",
        "It’s a passable product.",
        "Doesn’t stand out, but works fine.",
        "This is pretty standard.",
        "Nothing remarkable here.",
        "It’s just okay.",
        "The flight was fine, nothing special.",
        "I had no issues finding my gate for <MASKED_FLIGHT_NUM>.",
        "The PNR lookup process was okay but could be faster.",
        "The check-in process was standard, no complaints.",
        "I received the flight status updates as expected.",
        "The crew was neither great nor bad, just okay.",
        "My experience was average, nothing stood out.",
        "Boarding took longer than expected, but it was manageable.",
        "The flight wasn’t delayed, but it wasn’t early either.",
        "I got the information I needed, nothing more, nothing less.",
        "The luggage arrived as expected, no issues.",
        "I booked my flight with ease, but the UI could improve.",
        "The flight was on time, but the announcements were unclear.",
        "No major complaints, but no praises either.",
        "The service met my expectations, nothing exceeded them.",
        "It was an acceptable flight, nothing extraordinary.",
        "The seating was as I expected for economy class.",
        "I received the gate information, but it was last minute.",
        "The updates about flight <MASKED_FLIGHT_NUM> were sufficient.",
        "Check-in staff was polite but not particularly engaging.",
    ]
negative_samples = [
        "My flight was delayed by several hours. Terrible experience!",
        "The staff was rude and unhelpful during the flight.",
        "I didn’t get my preferred meal option even after requesting.",
        "The seats were cramped and uncomfortable for the entire journey.",
        "The boarding process was chaotic and poorly managed.",
        "My luggage got misplaced, and I’m still waiting for an update.",
        "The in-flight entertainment system was broken the whole time.",
        "I’m very disappointed with how my complaints were handled.",
        "The flight was cancelled without prior notice. Horrible service!",
        "I had a bad experience with the customer support team."
        "I’m very unhappy with the delayed flight <MASKED_FLIGHT_NUM>.",
        "My luggage didn’t arrive, and I’m frustrated.",
        "It took forever to find updates for my PNR <MASKED_PNR>.",
        "The customer service team was unhelpful and rude.",
        "I missed my connecting flight because of the delay.",
        "The flight was cancelled without prior notice.",
        "Rebooking for flight <MASKED_FLIGHT_NUM> was a nightmare.",
        "I’m extremely disappointed with how my complaint was handled.",
        "The boarding process was chaotic and disorganized.",
        "I waited too long to get information about my flight.",
        "The app kept crashing while I tried to check my PNR.",
        "I can’t believe how poor the in-flight service was.",
        "The delay for flight <MASKED_FLIGHT_NUM> ruined my travel plans.",
        "I’m upset about the lack of clear communication about cancellations.",
        "It’s unacceptable how long it took to get updates on my gate.",
        "The airline lost my luggage, and I still haven’t heard back.",
        "I regret choosing this airline for my trip.",
        "The crew was dismissive and unhelpful during the flight.",
        "The food options were terrible on flight <MASKED_FLIGHT_NUM>.",
        "No explanation was given for the delay. Very disappointing!",
        "This is a terrible product.",
        "I’m very disappointed in this.",
        "Horrible quality, avoid this.",
        "This was a waste of money.",
        "I regret buying this.",
        "Completely unreliable.",
        "This broke after a week of use.",
        "Worst experience ever!",
        "I wouldn’t recommend this to anyone.",
        "Absolutely frustrating to deal with.",
        "The quality is far below expectations.",
        "This is not worth the price.",
        "A complete disaster!",
        "This failed me when I needed it most.",
        "Don’t bother with this.",
        "Such a waste of time.",
        "This didn’t work as promised.",
        "I can’t believe I bought this.",
        "Extremely dissatisfied.",
        "This is a nightmare to use.",
        "I wish I could give this zero stars.",
        "Do not recommend to anyone.",
        "The worst purchase I’ve ever made.",
        "Nothing but problems with this.",
        "This was an awful experience.",
    ]
# Function to generate airline-specific sentiment dataset
def generate_airline_sentiment_dataset(output_file, samples_per_category=500):
    # Generate random samples
    positive = random.choices(positive_samples, k=samples_per_category // 3)
    neutral = random.choices(neutral_samples, k=samples_per_category // 3)
    negative = random.choices(negative_samples, k=samples_per_category // 3)
    # Combine all samples with their respective labels
    dataset = []
    for sample in positive:
        dataset.append([sample, "positive"])
    for sample in neutral:
        dataset.append([sample, "neutral"])
    for sample in negative:
        dataset.append([sample, "negative"])
    
    # Write to CSV
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])  # Column headers
        writer.writerows(dataset)
    
    print(f"Dataset successfully created with {samples_per_category} samples in {output_file}.")

# Generate dataset
generate_airline_sentiment_dataset("airline_sentiment_dataset.csv", 500)