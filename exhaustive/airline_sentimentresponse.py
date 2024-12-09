import pandas as pd
import random

# Expanded response templates for different sentiments
positive_responses = [
    "Thank you so much for your kind words! We're happy you had a great experience with Ace Airlines and look forward to welcoming you aboard again soon.",
    "We're thrilled to hear you had a positive experience with Ace Airlines. Your feedback is greatly appreciated, and we look forward to serving you again!",
    "Thank you for choosing Ace Airlines! We're glad you enjoyed your flight and we can't wait to welcome you back soon.",
    "We truly appreciate your feedback! It's always a pleasure to know our customers had a wonderful experience with us. See you on your next flight!",
    "Thank you for your wonderful feedback! We're delighted that you had a great experience with Ace Airlines and hope to serve you again soon!"
]

negative_responses = [
    "We sincerely apologize for your experience. Your feedback is valuable, and we'll work hard to ensure a better experience in the future.",
    "We're really sorry to hear about your experience. Your comments will be carefully reviewed to improve our service and we hope you give us another chance.",
    "We truly regret that your experience didn't meet your expectations. Please accept our sincerest apologies, and we will strive to make things better.",
    "We apologize for the inconvenience you faced. Your feedback is important, and we will take the necessary steps to improve.",
    "We are sorry for any issues you encountered with our service. We're committed to doing better and hope you'll consider flying with us again."
]

neutral_responses = [
    "Thank you for flying with Ace Airlines. We hope to exceed your expectations on your next flight!",
    "Thank you for choosing Ace Airlines. We're glad to hear that your flight was satisfactory, and we look forward to improving your next experience.",
    "We appreciate your feedback and thank you for flying with us. We strive to provide a better experience on every flight!",
    "Thank you for your comments. While we are glad your experience was okay, we aim to make every flight memorable. We hope to see you again soon!",
    "We appreciate your feedback and thank you for flying with Ace Airlines. We look forward to serving you better in the future."
]

positive_templates = [
    "I had an amazing experience with Ace Airlines!",
    "Ace Airlines made my trip so comfortable, thank you!",
    "Thank you, Ace Airlines, for the wonderful service!",
    "Absolutely loved my flight with Ace Airlines!",
    "Ace Airlines was amazing, I would definitely fly again!",
    "Great experience with Ace Airlines!",
    "The service from Ace Airlines was exceptional!",
    "Ace Airlines made my travel stress-free.",
    "The flight was perfect, couldn't ask for more!",
    "I love Ace Airlines, the best airline ever!",
    "I am so impressed with Ace Airlines’ service!",
    "Such a fantastic flight with Ace Airlines!",
    "I can't stop recommending Ace Airlines!",
    "A great experience all around with Ace Airlines!",
    "Ace Airlines always delivers the best service!",
    "Perfect flight experience with Ace Airlines!",
    "I had an enjoyable time on my Ace Airlines flight!",
    "Ace Airlines always exceeds my expectations!",
    "My flight with Ace Airlines was so smooth!",
    "Ace Airlines is my favorite airline by far!",
    "The crew was amazing, I felt so welcomed by Ace Airlines.",
    "Ace Airlines really knows how to make their customers happy!",
    "Thank you, Ace Airlines, for making my trip memorable!",
    "I was so comfortable throughout the flight with Ace Airlines.",
    "Ace Airlines is top-notch!",
    "Everything about my flight with Ace Airlines was excellent!",
    "The in-flight experience was superb with Ace Airlines.",
    "Ace Airlines made my journey so enjoyable!",
    "I’m always satisfied with Ace Airlines.",
    "Best flying experience I’ve ever had with Ace Airlines!",
    "Ace Airlines is always on time, and I love that!",
    "I'm a loyal customer of Ace Airlines and always will be!",
    "Ace Airlines has become my go-to airline!",
    "The service at Ace Airlines was fantastic!",
    "I had no complaints during my flight with Ace Airlines.",
    "Ace Airlines is my number one choice for flying.",
    "Such a comfortable flight with Ace Airlines!",
    "I couldn't be happier with my Ace Airlines flight!",
    "The staff at Ace Airlines is always so friendly!",
    "I love how easy it is to book with Ace Airlines!",
    "Ace Airlines takes customer service seriously.",
    "Fantastic experience with Ace Airlines every time!",
    "The amenities on board Ace Airlines were excellent.",
    "Always a pleasure flying with Ace Airlines!",
    "Ace Airlines’ flights are always comfortable and timely.",
    "The whole process with Ace Airlines was seamless!",
    "I trust Ace Airlines for all my travels!",
    "The entertainment system on board Ace Airlines was great!",
    "Ace Airlines is known for its punctuality, and they never disappoint!",
    "I feel well taken care of with Ace Airlines.",
    "Ace Airlines’ staff goes above and beyond.",
    "I will definitely choose Ace Airlines again!",
    "Ace Airlines always makes me feel at home when I fly.",
    "I had the best in-flight meal with Ace Airlines!",
    "I was so impressed by how clean the plane was.",
    "Ace Airlines is an outstanding airline.",
    "The customer service on board Ace Airlines was excellent!",
    "I appreciate how Ace Airlines takes care of its passengers.",
    "Flying with Ace Airlines is always a smooth experience.",
    "Ace Airlines is always a pleasant experience.",
    "I enjoy the frequent flyer benefits with Ace Airlines.",
    "Ace Airlines always goes the extra mile to make me comfortable.",
    "Ace Airlines is the gold standard in air travel!",
    "Such a pleasant journey with Ace Airlines!",
    "My flight with Ace Airlines was so smooth, no delays!",
    "I love how Ace Airlines keeps me updated about my flight.",
    "Ace Airlines never fails to impress me.",
    "The flight crew at Ace Airlines is always so accommodating.",
    "Flying with Ace Airlines is always a stress-free experience.",
    "I have never had a bad experience with Ace Airlines.",
    "Ace Airlines makes flying a pleasure!",
    "I will continue to choose Ace Airlines for my travel.",
    "I always feel safe and comfortable with Ace Airlines.",
    "The flight attendants were wonderful on my Ace Airlines flight.",
    "The flight was so smooth, Ace Airlines really knows how to do it right.",
    "Ace Airlines really focuses on the little details that make a difference.",
    "I love how Ace Airlines values its customers.",
    "I felt relaxed and well taken care of with Ace Airlines.",
    "Great experience with Ace Airlines, once again!",
    "I was able to rest and enjoy the flight thanks to Ace Airlines.",
    "Ace Airlines truly offers world-class service!",
    "I highly recommend Ace Airlines for any traveler!",
    "Ace Airlines always makes my travel experience better.",
    "Flying with Ace Airlines was so easy and hassle-free.",
    "Ace Airlines’ staff was exceptionally helpful and kind.",
    "I always feel welcomed aboard Ace Airlines flights.",
    "Ace Airlines never disappoints with its exceptional service.",
    "The amenities on my flight with Ace Airlines were fantastic!",
    "I enjoy how Ace Airlines makes flying a pleasant experience.",
    "Flying with Ace Airlines is always a treat!",
    "Ace Airlines made my trip so much more enjoyable.",
    "Best service in the sky, thank you Ace Airlines!",
    "I will continue to choose Ace Airlines for my future travels!",
    "I love how efficient Ace Airlines is!",
    "Flying with Ace Airlines was a delightful experience!",
    "I always look forward to flying with Ace Airlines.",
    "Ace Airlines is always on top of its game!",
    "I had a wonderful experience from start to finish with Ace Airlines.",
    "I love the in-flight experience with Ace Airlines.",
    "Ace Airlines is truly a class above the rest!",
    "Every time I fly with Ace Airlines, it’s a great experience.",
    "The crew on my Ace Airlines flight made everything so easy."
]

negative_templates = [
    "My flight with Ace Airlines was terrible!",
    "I had a horrible experience with Ace Airlines.",
    "The service from Ace Airlines was really bad.",
    "I'm very disappointed with Ace Airlines.",
    "Ace Airlines ruined my travel experience.",
    "My flight was delayed and nobody informed me!",
    "The flight attendants were rude and unhelpful.",
    "I had a horrible experience checking in with Ace Airlines.",
    "Ace Airlines lost my luggage and didn’t help me.",
    "The flight was very uncomfortable with Ace Airlines.",
    "I had a terrible time trying to reach customer service.",
    "Ace Airlines made a huge mistake with my booking.",
    "I’ll never fly with Ace Airlines again after that experience.",
    "The crew was unprofessional and didn’t care about passengers.",
    "My flight was overbooked, and I was left stranded.",
    "Ace Airlines didn’t follow up on my complaint.",
    "I had an awful experience with the food and drinks.",
    "The flight was so delayed, I missed my connecting flight.",
    "I had to wait forever to get any assistance from Ace Airlines.",
    "The overall experience with Ace Airlines was very frustrating.",
    "My seat was broken, and they did nothing to fix it.",
    "I had to deal with a lot of confusion at the airport.",
    "The flight was noisy and uncomfortable.",
    "My bag was damaged during the flight.",
    "The boarding process with Ace Airlines was chaotic.",
    "I felt completely neglected by the flight crew.",
    "The flight was delayed for hours without explanation.",
    "Ace Airlines messed up my reservation and didn’t help me.",
    "I received no apology or compensation for the issues.",
    "The aircraft was old and very uncomfortable.",
    "The customer service was appalling and unhelpful.",
    "I will never use Ace Airlines again after this experience.",
    "The flight attendants ignored me when I asked for help.",
    "I had a very negative experience with Ace Airlines.",
    "I can’t believe how bad my experience with Ace Airlines was.",
    "I was very disappointed with how Ace Airlines handled my issue.",
    "I had no communication from Ace Airlines about my delay.",
    "My flight with Ace Airlines was the worst I’ve ever had.",
    "The baggage claim process was a disaster.",
    "I was charged extra fees without any clear explanation.",
    "My flight was canceled without any notice.",
    "I had to deal with so many issues and nobody helped me.",
    "The flight was delayed for hours without any updates.",
    "I had a terrible time with Ace Airlines’ customer support.",
    "My experience with Ace Airlines was a total nightmare.",
    "I was treated poorly by the staff on my flight.",
    "Ace Airlines does not value its customers at all.",
    "I had a miserable time trying to resolve my issue with Ace Airlines.",
    "I would never recommend Ace Airlines to anyone.",
    "My seat was uncomfortable and there were no blankets.",
    "The check-in process was confusing and frustrating.",
    "I had issues with the Wi-Fi and no one helped me.",
    "My flight with Ace Airlines was full of delays and problems.",
    "The quality of service was not up to my expectations.",
    "I felt completely ignored by the staff throughout the flight.",
    "The flight was very uncomfortable and cramped.",
    "I had problems with the online booking system.",
    "There were no apologies or compensation for the inconvenience.",
    "The staff was rude and unaccommodating during my flight.",
    "Ace Airlines’ customer service was extremely unhelpful.",
    "I will never fly with Ace Airlines again after this incident.",
    "The flight was canceled at the last minute without explanation.",
    "The airline caused unnecessary stress during my trip.",
    "I had to deal with constant confusion at the airport.",
    "My luggage didn’t arrive on time and no one cared.",
    "Ace Airlines made me miss my connecting flight!",
    "The service on board was very poor.",
    "I’m very unhappy with the way Ace Airlines handled my complaint.",
    "Ace Airlines failed to meet basic expectations during my flight.",
    "I had to wait for hours just to talk to someone about my problem.",
    "The flight was delayed and nobody took responsibility for it.",
    "The entire experience with Ace Airlines was a disaster.",
    "My flight was an absolute nightmare.",
    "Ace Airlines ruined my travel plans."
]
neutral_templates = [
    "The flight was fine, but nothing extraordinary.",
    "I had a neutral experience with Ace Airlines.",
    "The flight was okay, just average.",
    "Ace Airlines was fine, nothing stood out.",
    "The service was alright, no complaints but not impressive.",
    "It was a standard flight with Ace Airlines.",
    "The experience was just average, nothing to rave about.",
    "Ace Airlines did what they had to, but I expected more.",
    "The flight was okay, but not as good as expected.",
    "I didn’t have any major complaints, but nothing was special.",
    "It wasn’t the best flight, but it wasn’t the worst either.",
    "The flight was fine, just didn’t exceed expectations.",
    "Ace Airlines met my expectations, but didn't go beyond them.",
    "It was a regular flight, nothing exceptional.",
    "I didn’t encounter any issues, but it didn’t stand out.",
    "The flight was uneventful, which is good in some ways.",
    "The service was fine, but could be better.",
    "I didn’t face any issues, but I wasn’t impressed either.",
    "The experience was standard, not memorable.",
    "The flight was smooth, but a bit forgettable.",
    "Everything went well, but nothing was outstanding.",
    "I didn’t have any major issues with my flight.",
    "Ace Airlines provided the basic service, but not much more.",
    "The flight was nothing special, but it was fine.",
    "I didn’t experience any major issues, but it was just okay.",
    "The service was adequate, but nothing exceptional.",
    "The flight was decent, but not remarkable.",
    "There was nothing wrong with the flight, but nothing memorable.",
    "Ace Airlines was fine, but I’ve had better experiences elsewhere.",
    "It was an average experience with Ace Airlines.",
    "I didn’t have any complaints, but also nothing to praise.",
    "The flight was typical, no major problems.",
    "The service was standard, nothing that stood out.",
    "The flight was neither good nor bad, just okay.",
    "The experience was decent, but nothing special.",
    "The flight was smooth, but it didn't exceed expectations.",
    "Everything went according to plan, but nothing exceptional happened.",
    "Ace Airlines met the basic expectations but didn’t go beyond.",
    "It was a neutral experience, nothing to complain about.",
    "The flight was normal, but lacked any special touches.",
    "I didn’t experience any problems, but it wasn’t outstanding.",
    "The experience was fine, but could have been better.",
    "It was a routine flight with no surprises.",
    "The flight was pleasant, but nothing remarkable.",
    "Nothing went wrong, but nothing stood out either.",
    "It was just an ordinary flight with Ace Airlines.",
    "The service was fine, but not up to my expectations.",
    "The flight was smooth but nothing exciting.",
    "I had no complaints, but it didn’t exceed expectations.",
    "I had no problems with my flight, but it was just average.",
    "The flight was comfortable, but lacked any wow factor.",
    "The service was fine, but could be more personal.",
    "The experience was okay, but I expected a bit more.",
    "The flight was smooth, but didn’t exceed my expectations.",
    "Nothing remarkable happened, just an average experience.",
    "The flight was alright, but could use some improvement.",
    "The experience was okay, but there’s room for improvement.",
    "Ace Airlines did what they had to, but it wasn’t extraordinary.",
    "It was a typical flight, nothing special.",
    "The flight was fine, but not impressive enough.",
    "I didn’t have any major issues, but it was an average flight.",
    "Everything was fine, but nothing extraordinary.",
    "The experience was decent, but lacked any excitement."
]

positive_responses = [
    "Thank you so much for your kind words! We're happy you had a great experience with Ace Airlines and look forward to welcoming you aboard again soon.",
    "We're thrilled to hear you had a positive experience with Ace Airlines. Your feedback is greatly appreciated, and we look forward to serving you again!",
    "Thank you for choosing Ace Airlines! We're glad you enjoyed your flight and we can't wait to welcome you back soon.",
    "We truly appreciate your feedback! It's always a pleasure to know our customers had a wonderful experience with us. See you on your next flight!",
    "Thank you for your wonderful feedback! We're delighted that you had a great experience with Ace Airlines and hope to serve you again soon!",
    "Your feedback is appreciated! We’re glad you had a great time flying with Ace Airlines. We hope to make your next journey even better.",
    "Thanks for flying with us! We’re so pleased to hear that you enjoyed your time with Ace Airlines. We hope to see you on your next trip!",
    "Thank you for sharing your experience! We’re thrilled to hear that you enjoyed your flight with Ace Airlines.",
    "We love hearing about your great experience! Thank you for choosing Ace Airlines, and we hope to welcome you back soon!",
    "It’s great to hear that you had a positive experience with Ace Airlines! We appreciate your feedback and hope to continue exceeding your expectations.",
    "Thanks for the great feedback! We love hearing how happy our passengers are with Ace Airlines. Looking forward to serving you again!",
    "We're so glad you enjoyed your flight! Thank you for choosing Ace Airlines, and we look forward to your next journey with us.",
    "Your positive feedback means a lot to us! We’re glad to know you had a pleasant flight with Ace Airlines. See you again soon!",
    "It’s always a pleasure to hear such positive feedback! Thank you for flying with Ace Airlines, and we can’t wait to serve you again!",
    "We appreciate you taking the time to share your positive experience! We’re thrilled to know you had a great time with Ace Airlines.",
    "Thank you for your kind words! We’re glad we could provide you with a pleasant experience, and we hope to welcome you back soon.",
    "We’re so happy to hear you enjoyed your flight! Thank you for your feedback, and we hope to continue delivering great service.",
    "Thank you for the kind review! We’re glad you had an excellent experience flying with Ace Airlines, and we look forward to your next flight.",
    "Your positive experience means the world to us! Thank you for choosing Ace Airlines, and we hope to see you again soon.",
    "We truly appreciate your feedback! It’s always our goal to deliver excellent service, and we’re glad to have met your expectations.",
    "Thank you for sharing your feedback! We’re happy to know you had a great experience with Ace Airlines and hope to serve you again soon.",
    "Thanks so much for your positive comments! We hope to make your next flight even better than the last.",
    "We are so happy you had a great experience! Thank you for flying with Ace Airlines, and we look forward to your next journey.",
    "Your feedback is invaluable! We’re thrilled that you had such a great experience with Ace Airlines. We look forward to seeing you again soon.",
    "Thank you for your amazing feedback! We’re glad you enjoyed your experience with Ace Airlines and hope to serve you again soon.",
    "We’re so pleased to hear that you enjoyed your flight! Thank you for sharing your experience with Ace Airlines.",
    "It’s always a pleasure to hear that our passengers are happy! Thank you for choosing Ace Airlines, and we look forward to flying with you again.",
    "We appreciate your kind words! Thank you for your support, and we can’t wait to welcome you aboard again soon.",
    "Thank you for your wonderful review! We’re so happy to hear that you had a fantastic flight with Ace Airlines.",
    "We truly appreciate your feedback! We’re happy you had a wonderful experience with us and hope to serve you again soon.",
    "Thank you for taking the time to leave a review! We’re glad to hear that Ace Airlines made your trip a pleasant one.",
    "Your feedback helps us improve our service! Thank you for sharing your positive experience with Ace Airlines.",
    "We’re so pleased to hear you had a positive experience with Ace Airlines! Your feedback is important to us, and we look forward to serving you again.",
    "Thank you for your amazing review! We’re glad to hear that Ace Airlines met your expectations and hope to see you again soon.",
    "Thanks for the kind words! We love to hear that our passengers are happy, and we hope to serve you again soon.",
    "Your satisfaction is our priority! Thank you for flying with Ace Airlines and sharing your positive experience.",
    "We’re so grateful for your feedback! Thank you for choosing Ace Airlines, and we look forward to making your next trip even better.",
    "It’s wonderful to hear that you had a great flight! Thank you for flying with Ace Airlines, and we can’t wait to welcome you back.",
    "Your feedback brightens our day! We’re thrilled to know you had a fantastic experience with Ace Airlines. Looking forward to your next journey.",
    "Thank you for your positive review! We’re so pleased that Ace Airlines made your trip enjoyable. We can’t wait to serve you again.",
    "We love hearing about our passengers’ positive experiences! Thank you for flying with Ace Airlines, and we hope to see you on your next trip.",
    "Your satisfaction is our reward! We’re so happy you had a pleasant experience with Ace Airlines. We look forward to serving you again soon.",
    "Thank you for sharing your wonderful experience! We’re thrilled that you enjoyed your flight with Ace Airlines, and we hope to see you again soon.",
    "Thanks for the great feedback! It’s always a pleasure to hear that our passengers had a positive experience with Ace Airlines.",
    "We’re so happy you had a great experience! Thank you for flying with Ace Airlines, and we look forward to welcoming you back soon.",
    "Your feedback is so appreciated! We’re glad to know that Ace Airlines made your journey a pleasant one. See you again soon!",
    "Thank you for your kind words! We’re thrilled to know you had a great flight with Ace Airlines.",
    "We’re delighted to hear that you had a wonderful experience! Thank you for choosing Ace Airlines for your journey.",
    "Thank you for sharing your great experience! We’re so pleased to know that Ace Airlines made your trip enjoyable.",
    "We love hearing positive feedback! Thank you for choosing Ace Airlines, and we look forward to your next flight with us.",
    "Thank you for your feedback! We’re so happy that Ace Airlines met your expectations, and we hope to see you again soon.",
    "Thank you for your support! We’re glad you enjoyed your flight and look forward to providing you with an even better experience next time.",
    "We are so pleased to hear that you had a great experience with Ace Airlines! Thank you for your valuable feedback."
]
negative_responses = [
    "We sincerely apologize for your experience. Your feedback is valuable, and we'll work hard to ensure a better experience in the future.",
    "We're really sorry to hear about your experience. Your comments will be carefully reviewed to improve our service and we hope you give us another chance.",
    "We truly regret that your experience didn't meet your expectations. Please accept our sincerest apologies, and we will strive to make things better.",
    "We apologize for the inconvenience you faced. Your feedback is important, and we will take the necessary steps to improve.",
    "We are sorry for any issues you encountered with our service. We're committed to doing better and hope you'll consider flying with us again.",
    "We truly regret the difficulties you faced. Please reach out to our customer service team, and we'll ensure your next experience is much better.",
    "Our sincerest apologies for the trouble you encountered. We’ll investigate and work hard to make sure this doesn’t happen again.",
    "We’re sorry to hear about the challenges you faced during your flight. Rest assured, we’ll work on improving our services based on your feedback.",
    "We deeply apologize for your experience. Please know we’re actively working on making improvements to ensure better service in the future.",
    "We're so sorry you had a negative experience. Your feedback helps us identify areas to improve, and we’re committed to doing better.",
    "Our sincerest apologies for any inconvenience caused. We’ll address your concerns and work to make your next experience better.",
    "We are truly sorry for your experience with us. Please accept our apology, and we hope you’ll consider giving us another chance in the future.",
    "We regret hearing about your difficulties with Ace Airlines. Your concerns are being reviewed so we can improve.",
    "We apologize for your experience and are working diligently to address the issues you raised. Your feedback is invaluable to us.",
    "We’re sorry to hear that your flight didn’t meet your expectations. We’ll ensure that these issues are addressed to improve our service.",
    "We apologize for the poor experience. We take your feedback seriously, and we’ll ensure this is resolved for future flights.",
    "Please accept our apologies for your negative experience. We're committed to improving and hope to serve you better next time.",
    "We are very sorry for your unsatisfactory experience. Your feedback will help us improve our services.",
    "We apologize for the issues you encountered. Please allow us another chance to provide you with a better experience next time.",
    "We're truly sorry that you had a bad experience. Rest assured that we’ll work to fix these issues moving forward.",
    "We’re very sorry to hear that you had such an unpleasant experience. Your concerns are being addressed, and we’re committed to doing better.",
    "We regret that your experience with us didn’t meet expectations. We’re taking the necessary steps to improve for future passengers.",
    "We apologize for the inconvenience. Your feedback is being taken seriously, and we’ll make sure to provide better service next time.",
    "We are sorry to hear about your recent experience. We’re working on ensuring that this doesn’t happen again.",
    "We sincerely apologize for your negative experience. Our team is addressing this issue to provide a more comfortable experience next time.",
    "We regret hearing about your issues with our service. We're actively working to improve and provide a better experience next time.",
    "We're sorry you had to deal with this. Please reach out to our support team for assistance, and we’ll ensure your concerns are resolved.",
    "We’re truly sorry for your experience, and we assure you that we’re working to make necessary improvements.",
    "We are very sorry that you encountered issues during your flight. We’ll investigate and ensure that this is resolved for future flights.",
    "Our sincerest apologies for the trouble you encountered. We're working hard to improve our service, and we hope you'll give us another chance.",
    "We’re sorry to hear you had such a negative experience. We’ll take immediate action to resolve the issues you mentioned.",
    "Our apologies for any inconvenience. Your feedback helps us improve, and we will ensure better service going forward.",
    "We regret that we didn’t meet your expectations. Please accept our apology, and we will work to address the issues you mentioned.",
    "We apologize for any delays or issues you faced. Rest assured, we are working on improving our service.",
    "We regret hearing about your unsatisfactory experience. We’re investigating to ensure this doesn’t happen again.",
    "We’re sorry your flight experience wasn't up to par. We're reviewing your feedback and will take the necessary steps to improve.",
    "We truly regret that your experience didn’t meet expectations. Please know that we take your concerns seriously and will work to improve.",
    "We sincerely apologize for your negative experience. We’ll take your feedback into account as we make improvements.",
    "We’re sorry that we fell short of expectations. Please let us know how we can make it right and ensure a better experience next time.",
    "Our sincerest apologies for the issues you experienced. We’ll address your concerns and work to improve.",
    "We regret the issues you encountered. Your feedback is crucial to us, and we will take immediate steps to improve.",
    "We truly apologize for the inconvenience caused. We’ll investigate your feedback and ensure improvements are made."
]
neutral_responses = [
    "Thank you for flying with Ace Airlines. We hope to exceed your expectations on your next flight!",
    "Thank you for choosing Ace Airlines. We're glad to hear that your flight was satisfactory, and we look forward to improving your next experience.",
    "We appreciate your feedback and thank you for flying with us. We strive to provide a better experience on every flight!",
    "Thank you for your comments. While we are glad your experience was okay, we aim to make every flight memorable. We hope to see you again soon!",
    "We appreciate your feedback and thank you for flying with Ace Airlines. We look forward to serving you better in the future.",
    "Thank you for your feedback! We’re glad you had an okay experience and hope to improve our service even more for your next flight.",
    "Thanks for sharing your experience. We’re happy your flight went smoothly, and we look forward to making your next journey with us even better.",
    "We appreciate your feedback and thank you for flying with us. Your satisfaction is important to us, and we hope to exceed your expectations next time.",
    "Thank you for your comments! We strive to provide the best experience for all our passengers and look forward to making your next flight even better.",
    "Thanks for flying with Ace Airlines! We’re happy to hear you had a satisfactory flight and we hope to continue improving our service for you.",
    "We’re glad your flight was satisfactory. Thank you for your feedback, and we hope to offer an even better experience next time.",
    "Thank you for sharing your experience. While we always aim for perfection, we appreciate your thoughts and hope to serve you even better next time.",
    "Thanks for flying with us. We hope your next experience with Ace Airlines will be even more enjoyable.",
    "We value your feedback. We’re glad you had a satisfactory flight, and we’ll continue working to improve our service.",
    "Thank you for your feedback! We hope to exceed your expectations with every flight. See you soon!",
    "Thanks for sharing your thoughts. We’re happy your experience was satisfactory and hope to make future flights even better.",
    "We appreciate your feedback! While we know there’s always room for improvement, we’re glad your experience was acceptable.",
    "Thank you for flying with us. We strive to improve with every flight, and we look forward to serving you again soon.",
    "Thank you for your feedback! We hope to deliver a better experience on your next flight with Ace Airlines.",
    "We value your feedback! We strive to make each flight better and hope your next one with us will be even more enjoyable.",
    "Thanks for sharing your experience! We’re happy you had a smooth flight and hope to serve you even better next time.",
    "We appreciate your feedback! While we know there’s always room for improvement, we’re glad your experience was satisfactory.",
    "Thanks for flying with us! Your feedback helps us improve, and we look forward to making your next experience even better.",
    "Thank you for your feedback. We’re glad you had a satisfactory flight and look forward to serving you even better next time.",
    "We appreciate your feedback! We hope your next experience with Ace Airlines will be even more enjoyable.",
    "Thank you for your comments! We value your input and will continue to strive for a better experience next time.",
    "Thanks for sharing your feedback. We appreciate your thoughts and hope your next experience with us will be even better.",
    "Thank you for flying with Ace Airlines. We’re glad your experience was acceptable, and we hope to improve our service on your next flight.",
    "We appreciate your feedback! Thank you for choosing Ace Airlines, and we look forward to serving you again.",
    "We’re glad you had a satisfactory experience! Your feedback is important to us, and we look forward to improving your next flight.",
    "Thank you for your comments. We’re glad to hear your experience was fine, and we’re excited to provide an even better experience next time.",
    "Thanks for your feedback! We hope your next experience with Ace Airlines will exceed your expectations.",
    "We appreciate you flying with Ace Airlines. Your feedback helps us improve, and we hope to serve you better next time.",
    "Thank you for your feedback! We hope to make your next flight experience with us even better.",
    "We appreciate your comments! We’re glad you had a satisfactory flight and look forward to your next one.",
    "Thanks for your feedback! We strive for a great experience with every flight, and we hope your next flight will be even better.",
    "We appreciate your feedback! While we always aim for excellence, we’re glad your experience was acceptable.",
    "Thanks for flying with us! We hope your next flight experience will be even more enjoyable.",
    "Thank you for your feedback! We’re glad you had a good experience with Ace Airlines and hope to serve you better in the future.",
    "Thank you for your comments. We hope to serve you better in the future and deliver an even better experience next time.",
    "Thanks for flying with us! We hope your next experience with Ace Airlines is even more delightful.",
    "Thank you for sharing your thoughts with us. We appreciate your feedback and hope to improve your experience on future flights.",
    "We appreciate your feedback! Thanks for flying with Ace Airlines, and we look forward to providing an even better service next time."
]


# Generate dataset with 500 unique samples
data = []

# For each sentiment (positive, negative, neutral), generate multiple questions and responses
for sentiment, templates, responses in [
    ("positive", positive_templates, positive_responses),
    ("negative", negative_templates, negative_responses),
    ("neutral", neutral_templates, neutral_responses)
]:
    for user_input in templates:
        # For each user_input (template), generate multiple responses
        for response in responses:
            data.append({"user_input": user_input, "sentiment": sentiment, "response": response})

# Convert to DataFrame for easy handling and export
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("sentiment_conversational_response.csv", index=False)

print("CSV Dataset with multiple responses generated and saved as 'sentiment_conversational_data_expanded.csv'.")