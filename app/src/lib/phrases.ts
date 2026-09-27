export type Role = "customer" | "staff";
export type Language = import("./speech").LangCode;
export type ServiceId =
  "general" | "banking" | "hospitality" | "retail" | "transport" | "public";
export type Phrase = {
  service?: ServiceId;
  id: string;
  role: Role;
  category: string;
  en: string;
  hi: string;
};
export const phrases: Phrase[] = [
  {
    id: "general-1",
    service: "general",
    role: "customer",
    category: "General services",
    en: "I need help with a service. Can you guide me?",
    hi: "मुझे एक सेवा के बारे में मदद चाहिए। क्या आप मेरा मार्गदर्शन कर सकते हैं?",
  },
  {
    id: "general-2",
    service: "general",
    role: "staff",
    category: "General services",
    en: "Of course. Please tell me what you need help with.",
    hi: "ज़रूर। कृपया बताइए कि आपको किस बारे में मदद चाहिए।",
  },
  {
    id: "general-3",
    service: "general",
    role: "customer",
    category: "General services",
    en: "Please explain the process and any charges in writing.",
    hi: "कृपया प्रक्रिया और किसी भी शुल्क के बारे में लिखकर समझाएँ।",
  },
  {
    id: "general-4",
    service: "general",
    role: "staff",
    category: "General services",
    en: "I can explain the process step by step. You can ask me to repeat anything.",
    hi: "मैं प्रक्रिया को एक-एक कदम करके समझा सकता हूँ। आप कोई भी बात दोबारा पूछ सकते हैं।",
  },
  {
    id: "hospitality-1",
    service: "hospitality",
    role: "customer",
    category: "Hospitality",
    en: "I have a reservation. Can you help me check in?",
    hi: "मेरी बुकिंग है। क्या आप चेक-इन करने में मेरी मदद कर सकते हैं?",
  },
  {
    id: "hospitality-2",
    service: "hospitality",
    role: "staff",
    category: "Hospitality",
    en: "Please show me your booking confirmation so I can help.",
    hi: "कृपया अपनी बुकिंग की पुष्टि दिखाएँ ताकि मैं मदद कर सकूँ।",
  },
  {
    id: "hospitality-3",
    service: "hospitality",
    role: "customer",
    category: "Hospitality",
    en: "Please write down the check-out time and any extra charges.",
    hi: "कृपया चेक-आउट का समय और अतिरिक्त शुल्क लिख दें।",
  },
  {
    id: "hospitality-4",
    service: "hospitality",
    role: "staff",
    category: "Hospitality",
    en: "I will explain the timings and charges before you proceed.",
    hi: "आगे बढ़ने से पहले मैं समय और शुल्क समझा दूँगा।",
  },
  {
    id: "retail-1",
    service: "retail",
    role: "customer",
    category: "Retail",
    en: "I would like to return an item. What are my options?",
    hi: "मैं एक सामान वापस करना चाहता हूँ। मेरे पास क्या विकल्प हैं?",
  },
  {
    id: "retail-2",
    service: "retail",
    role: "staff",
    category: "Retail",
    en: "Please show me the item and receipt so I can check the return policy.",
    hi: "कृपया सामान और रसीद दिखाएँ ताकि मैं वापसी की नीति जाँच सकूँ।",
  },
  {
    id: "retail-3",
    service: "retail",
    role: "customer",
    category: "Retail",
    en: "Can I get a refund or exchange? Please explain in writing.",
    hi: "क्या मुझे पैसे वापस मिल सकते हैं या सामान बदला जा सकता है? कृपया लिखकर समझाएँ।",
  },
  {
    id: "retail-4",
    service: "retail",
    role: "staff",
    category: "Retail",
    en: "I need to check the policy before confirming a refund or exchange.",
    hi: "पैसे वापस करने या सामान बदलने की पुष्टि से पहले मुझे नीति जाँचनी होगी।",
  },
  {
    id: "transport-1",
    service: "transport",
    role: "customer",
    category: "Transport & travel",
    en: "I need help planning my journey. Where should I go?",
    hi: "मुझे अपनी यात्रा की योजना बनाने में मदद चाहिए। मुझे कहाँ जाना चाहिए?",
  },
  {
    id: "transport-2",
    service: "transport",
    role: "staff",
    category: "Transport & travel",
    en: "Please tell me your destination and preferred travel time.",
    hi: "कृपया अपना गंतव्य और यात्रा का पसंदीदा समय बताइए।",
  },
  {
    id: "transport-3",
    service: "transport",
    role: "customer",
    category: "Transport & travel",
    en: "Please write down the departure point, time and fare.",
    hi: "कृपया प्रस्थान की जगह, समय और किराया लिख दें।",
  },
  {
    id: "transport-4",
    service: "transport",
    role: "staff",
    category: "Transport & travel",
    en: "I can help you check the route and fare before you book.",
    hi: "बुकिंग से पहले मैं मार्ग और किराया जाँचने में आपकी मदद कर सकता हूँ।",
  },
  {
    id: "public-1",
    service: "public",
    role: "customer",
    category: "Public services",
    en: "I need help with an application. What should I do first?",
    hi: "मुझे एक आवेदन में मदद चाहिए। मुझे सबसे पहले क्या करना चाहिए?",
  },
  {
    id: "public-2",
    service: "public",
    role: "staff",
    category: "Public services",
    en: "Please tell me which service you are applying for.",
    hi: "कृपया बताइए कि आप किस सेवा के लिए आवेदन कर रहे हैं।",
  },
  {
    id: "public-3",
    service: "public",
    role: "customer",
    category: "Public services",
    en: "Which documents do I need, and is there a fee?",
    hi: "मुझे कौन से दस्तावेज़ चाहिए और क्या कोई शुल्क है?",
  },
  {
    id: "public-4",
    service: "public",
    role: "staff",
    category: "Public services",
    en: "I can help you check the requirements and write down the next steps.",
    hi: "मैं आवश्यकताएँ जाँचने और अगले कदम लिखने में आपकी मदद कर सकता हूँ।",
  },
  {
    id: "charge",
    service: "banking",
    role: "customer",
    category: "Account charges",
    en: "I noticed an unexpected charge on my account. Can you explain it?",
    hi: "मेरे खाते में एक ऐसा शुल्क लगा है जिसकी मुझे उम्मीद नहीं थी। क्या आप इसे समझा सकते हैं?",
  },
  {
    id: "check",
    service: "banking",
    role: "staff",
    category: "Account charges",
    en: "I can help you check that charge. Please show me the date and amount on your statement.",
    hi: "मैं उस शुल्क की जाँच करने में आपकी मदद कर सकता हूँ। कृपया अपने विवरण में तारीख और राशि दिखाएँ।",
  },
  {
    id: "detail",
    service: "banking",
    role: "customer",
    category: "Account charges",
    en: "The charge is ₹250, dated 12 September. What is it for?",
    hi: "12 सितंबर को ₹250 का शुल्क लगा है। यह किस लिए है?",
  },
  {
    id: "review",
    service: "banking",
    role: "staff",
    category: "Account charges",
    en: "I need to review the charge before I can explain it. I can write down the next steps for you.",
    hi: "समझाने से पहले मुझे शुल्क की जाँच करनी होगी। मैं आपके लिए अगले कदम लिख सकता हूँ।",
  },
  {
    id: "dispute",
    service: "banking",
    role: "customer",
    category: "Account charges",
    en: "I do not recognise this charge. How can I raise a complaint?",
    hi: "मैं इस शुल्क को नहीं पहचानता हूँ। मैं शिकायत कैसे कर सकता हूँ?",
  },
  {
    id: "steps",
    service: "banking",
    role: "staff",
    category: "Account charges",
    en: "I can explain our complaint process and help you with the form.",
    hi: "मैं शिकायत की प्रक्रिया समझा सकता हूँ और फ़ॉर्म भरने में आपकी मदद कर सकता हूँ।",
  },
  {
    id: "thanks",
    role: "customer",
    category: "Everyday help",
    en: "Thank you. Please write down what I need to do next.",
    hi: "धन्यवाद। कृपया लिख दें कि मुझे आगे क्या करना है।",
  },
  {
    id: "repeat",
    role: "customer",
    category: "Communication",
    en: "Please explain that again in simpler words.",
    hi: "कृपया इसे फिर से आसान शब्दों में समझाएँ।",
  },
  {
    id: "write",
    role: "customer",
    category: "Communication",
    en: "I use Indian Sign Language. Please communicate with me in writing for now.",
    hi: "मैं भारतीय सांकेतिक भाषा का उपयोग करता हूँ। अभी कृपया मुझसे लिखकर बात करें।",
  },
  {
    id: "interpreter",
    role: "customer",
    category: "Communication",
    en: "Is an Indian Sign Language interpreter available?",
    hi: "क्या भारतीय सांकेतिक भाषा का दुभाषिया उपलब्ध है?",
  },
  {
    id: "time",
    role: "staff",
    category: "Communication",
    en: "Please take your time. You can type or choose a phrase here.",
    hi: "आराम से बताइए। आप यहाँ लिख सकते हैं या कोई वाक्य चुन सकते हैं।",
  },
  {
    id: "understand",
    role: "staff",
    category: "Everyday help",
    en: "Would you like me to explain anything else?",
    hi: "क्या आप चाहते हैं कि मैं कुछ और समझाऊँ?",
  },
];
export const phraseText = (phrase: Phrase, language: Language) =>
  language === "hi-IN" ? phrase.hi : phrase.en;

export const services: {
  id: ServiceId;
  label: string;
  title: string;
  steps: string[];
}[] = [
  {
    id: "general",
    label: "General services",
    title: "Getting the help you need",
    steps: ["general-1", "general-2", "general-3", "general-4", "thanks"],
  },
  {
    id: "banking",
    label: "Banking",
    title: "An unexpected account charge",
    steps: ["charge", "check", "detail", "review", "thanks"],
  },
  {
    id: "hospitality",
    label: "Hospitality",
    title: "Asking about a reservation",
    steps: [
      "hospitality-1",
      "hospitality-2",
      "hospitality-3",
      "hospitality-4",
      "thanks",
    ],
  },
  {
    id: "retail",
    label: "Retail",
    title: "Asking about a return",
    steps: ["retail-1", "retail-2", "retail-3", "retail-4", "thanks"],
  },
  {
    id: "transport",
    label: "Transport & travel",
    title: "Finding the right journey",
    steps: [
      "transport-1",
      "transport-2",
      "transport-3",
      "transport-4",
      "thanks",
    ],
  },
  {
    id: "public",
    label: "Public services",
    title: "Understanding an application",
    steps: ["public-1", "public-2", "public-3", "public-4", "thanks"],
  },
];
