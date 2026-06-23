# In-domain call test error analysis

Evaluated 1350 examples. Invalid structured outputs are treated as non-fraud, matching the main evaluator.

## Key findings

- Of 67 false negatives, **55** come from schema-invalid outputs; only 12 are valid low-risk verdicts.
- Missing required fields account for **64 / 71** invalid outputs.
- Invalid outputs are concentrated on fraud examples: **55 / 71** have a fraud gold label.
- 65 invalid outputs still contain a valid `risk` value. A diagnostic risk-only fallback would yield F1 **0.973**; this is not the reported production metric.
- One generic output accounts for **3 / 9** false positives (33.3%), indicating a repeated default-high response.

## LLM confusion matrix

| | predicted non-fraud | predicted fraud |
|---|---:|---:|
| gold non-fraud | 666 | 9 |
| gold fraud | 67 | 608 |

## Invalid structured outputs

Total: **71** (5.3%)

| reason | count |
|---|---:|
| invalid_enum | 1 |
| malformed_json | 6 |
| missing_required_field | 64 |

By gold label: fraud=55, non_fraud=16.

## Sample: false_positive (9 total)

| index | gold | pred | valid | failure | transcript excerpt | raw output |
|---:|---|---|---|---|---|---|
| 125 | False | True | True | — | Innocent: Hello. Suspect: Hi, this is Karen from Advanced Medical Associates. I'm calling to confirm your upcoming appointment with Dr. Smith on Thursday at 2 PM. Can you please c… | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 282 | False | True | True | — | Innocent: Hello. Suspect: Hi, this is Karen from Smith's Medical Group. I'm calling to confirm your appointment with Dr. Johnson on Friday at 2 PM. Can you confirm that you're sti… | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 290 | False | True | True | — | Innocent: Hello. Suspect: Hi there! My name is Rachel and I'm an insurance representative. How are you today? Innocent: Cut the small talk, Rachel. What do you want? Suspect: I ap… | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 316 | False | True | True | — | caller: Hi, this is Alex from AT&T support. I'm calling about an issue with your internet connection. receiver: What's the problem? My internet has been out since yesterday. calle… | {"risk": "high", "fraud_type": "tech_support_scam", "reason": "Caller exhibits a support scam pattern.", "flagged_spans": []} |
| 701 | False | True | True | — | caller: Hi, this is Officer Johnson from the Social Security Administration. I'm calling to verify some information about your account. receiver: Hi Officer Johnson, what kind of … | {"risk": "high", "fraud_type": "ssn_scam", "reason": "Caller exhibits a ssn scam pattern.", "flagged_spans": []} |
| 785 | False | True | True | — | caller: Hello, my name is Emily from Microsoft Support. We've received an alert that your Office 365 subscription is experiencing issues with syncing your calendar. Can you confir… | {"risk": "high", "fraud_type": "tech_support_scam", "reason": "Caller exhibits a support scam pattern.", "flagged_spans": []} |
| 831 | False | True | True | — | caller: Hello, is this John Lee? receiver: Yes, that's me. Who's calling? caller: I'm Officer Thompson from the Social Security Administration. We've received a report that your s… | {"risk": "high", "fraud_type": "ssn_scam", "reason": "Caller exhibits a ssn scam pattern.", "flagged_spans": []} |
| 1032 | False | True | True | — | Innocent: Hello. Suspect: Hi, this is Karen from Speedy Deliveries. I'm calling to confirm the details of your upcoming delivery. Can I please speak with Emily Wilson? Innocent: W… | {"risk": "high", "fraud_type": "refund_scam", "reason": "Caller exhibits a refund scam pattern.", "flagged_spans": []} |
| 1308 | False | True | True | — | caller: Hi, may I speak with Mr. Thompson, please? receiver: Yes, this is Mr. Thompson. How can I help you? caller: Mr. Thompson, my name is Officer Michael from the Social Securi… | {"risk": "high", "fraud_type": "ssn_scam", "reason": "Caller exhibits a ssn scam pattern.", "flagged_spans": []} |

## Sample: false_negative (67 total)

| index | gold | pred | valid | failure | transcript excerpt | raw output |
|---:|---|---|---|---|---|---|
| 66 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, this is David from TechShield Solutions. We've been monitoring your computer's activity and it appears that your system has been infected with a m… | {"risk": "high"} |
| 111 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello there, Mrs. Johnson! Congratulations are in order! You've been selected as a winner in our recent sweepstakes drawing. You've won a cash prize of $… | {"risk": "high", "confidence": "high"} |
| 200 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, this is John from TechShield Solutions. We've been monitoring your computer's activity and have detected a critical security breach. Your device h… | {"risk": "high"} |
| 271 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hi there, this is Karen from Apex Financial Services. How are you today? I'm calling to inform you that you're eligible for a refund of $427.89. Our syst… | {"risk": "high"} |
| 329 | True | False | True | — | caller: From school transport—bus pass expired. Renew with your DOB and card. receiver: 06/17/2009, card 4537 1902 8476 0033. caller: Pass is renewed. | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 374 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, this is Karen from National Billing Services. How are you today, Ms. Johnson? I'm calling to inform you that our system has detected an overpaymen… | {"risk": "high", "savings": "none"} |
| 386 | True | False | True | — | caller: Hello, this is Karen from Microsoft support. I'm calling about a recent update issue with your Windows operating system. receiver: Hi Karen, thanks for reaching out. I've … | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate support call.", "flagged_spans": []} |
| 401 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, my name is David from TechShield Solutions. I'm calling to inform you that our system has detected a malware infection on your computer. It's a ve… | {"risk": "high", "confidence": "medium"} Explanation: The caller exhibits a scam pattern, and the situation appears to be a support scam. There is a high risk involved because the… |
| 691 | True | False | False | malformed_json | Innocent: Hello. Suspect: Hello, my name is David from TechShield Solutions. We've been monitoring your internet activity and our system has detected a malware infection on your c… | {"risk": "high"} Explanation: The caller exhibits a scam pattern, a legitimate-sounding but ultimately fraudulent operation. They display a support scam pattern, a confidence scam… |
| 705 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, is this Mr. Johnson? My name is Officer Thompson from the Social Security Administration. I'm calling to inform you that we've received a notifica… | {"risk": "high"} Explanation: The caller exhibits a scam pattern, a legitimate refund call, and a support scam scenario. There is a high risk involved; therefore, no refund should… |
| 774 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello! My name is Karen Thompson and I'm calling from National Benefits Corporation. How are you today? Innocent: I'm doing well, thanks. What can Nation… | {"risk": "high", "suspicion": "very_high"} |
| 864 | True | False | True | — | caller: We found a problem with your health insurance coverage. To avoid loss of benefits, verify your policy number and DOB. receiver: Policy 589456771, DOB 07/14/1977. caller: W… | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 876 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, is this Emily Wilson? My name is Rachel from Account Services at American Expressions Inc. I'm calling to inform you that our system has detected … | {"risk": "high", "suspiciousness": "very_high"} |
| 892 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, this is Karen from Universal Services. How are you today? I'm calling to inform you that our system has flagged your account for a refund of $472.… | {"risk": "high"} |
| 897 | True | False | True | — | caller: Hi, this is Sarah from Amazon's premium support team. We're reaching out about a potential issue with your account security. receiver: Oh, hi Sarah. I'm not aware of any i… | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate support call.", "flagged_spans": []} |
| 1012 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, Mr. Balku. This is Officer Rachel Thompson from the Social Security Administration. We've been trying to reach you regarding a very serious matter… | {"risk": "high", "suspicion": "very_high"} |
| 1073 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, Mr. Icli. This is Officer Rachel Johnson from the Social Security Administration. We've been trying to reach you about a serious issue with your s… | {"risk": "high", "suspicion": "very_high"} |
| 1095 | True | False | True | — | caller: Hello, is this Mr. Johnson? This is a call from Visa customer service, and we've been trying to reach you about a suspicious transaction on your account. receiver: Oh, I w… | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate support call.", "flagged_spans": []} |
| 1213 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, is this Karen Wilson? My name is David Lee and I'm calling from United Refund Services. We've been trying to reach you regarding a refund you're o… | {"risk": "high", "suspicion": "very_high"} |
| 1277 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, this is Officer Johnson from the Social Security Administration. I'm calling about a very serious matter regarding your social security number. We… | {"risk": "high", "suspicion": "very_high"} |

## Sample: invalid_json (71 total)

| index | gold | pred | valid | failure | transcript excerpt | raw output |
|---:|---|---|---|---|---|---|
| 39 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello! My name is David and I'm calling from TechGuard Solutions. We've been monitoring your computer's activity and it appears that your system has been… | {"risk": "high"} |
| 77 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hi there, this is Rachel from TechSupport Solutions. How are you today? I'm calling to inform you that our system has detected an overpayment on your acc… | {"risk": "high", "suspiciousness": "very_high"} |
| 144 | False | False | False | missing_required_field | Innocent: Hello. Suspect: Hi there! My name is Rachel and I'm an insurance sales representative. How are you today? Innocent: I'm doing great, Rachel. Thanks for asking. What can … | {"risk": "low", "confidence": "high"} |
| 167 | False | False | False | missing_required_field | Innocent: Hello. Suspect: Hi there! My name is Alex and I'm an insurance representative. How are you today? Innocent: I'm a little nervous, to be honest. I've been having some iss… | {"risk": "low", "confidence": "high"} |
| 363 | False | False | False | missing_required_field | Innocent: Hello. Suspect: Hi there! My name is Rachel and I'm an insurance sales representative. How are you doing today? Innocent: Hi Rachel, I'm doing great, thanks! What can I … | {"risk": "low", "confidence": "high"} |
| 394 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, this is Alex from TechShield Solutions. We've been monitoring your computer's activity and have detected some suspicious malware infections. Your … | {"risk": "high", "scam_type": "tech_support_scam"} |
| 401 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, my name is David from TechShield Solutions. I'm calling to inform you that our system has detected a malware infection on your computer. It's a ve… | {"risk": "high", "confidence": "medium"} Explanation: The caller exhibits a scam pattern, and the situation appears to be a support scam. There is a high risk involved because the… |
| 487 | True | False | False | malformed_json | Innocent: Hello. Suspect: Hello, this is Officer Johnson from the Social Security Administration. We've been trying to reach you about a very serious matter. Your social security … | {"risk": "high", "confidence": "medium"} Explanation: The call exhibits a scam pattern, a legitimate-sounding refund or reward call, and a refund scam subpattern. The caller exhib… |
| 544 | False | False | False | missing_required_field | Innocent: Hello. Suspect: Hi there! My name is John and I'm an insurance sales representative. How are you doing today? Innocent: I think I'm doing okay. Who's John? Why are you c… | {"risk": "low", "type": "insurance"} |
| 584 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, this is David from TechShield Support. We've been monitoring your computer's activity and we've detected some suspicious malware that's been causi… | {"risk": "high", "suspicion": "very_high"} |
| 738 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, this is Alex from TechCare Solutions. We've been monitoring your computer's activity and detected a severe virus infection. It's spreading rapidly… | {"risk": "high"} Innocent appears to be dealing with a legitimate-looking scam. The pattern matches a support scam where a caller pretends to be from a tech support service. Such … |
| 822 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, my name is Alex from TechGuard Solutions. We've been monitoring your internet activity and our system has detected a malware infection on your com… | {"risk": "high"} |
| 892 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, this is Karen from Universal Services. How are you today? I'm calling to inform you that our system has flagged your account for a refund of $472.… | {"risk": "high"} |
| 1012 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, Mr. Balku. This is Officer Rachel Thompson from the Social Security Administration. We've been trying to reach you regarding a very serious matter… | {"risk": "high", "suspicion": "very_high"} |
| 1126 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hi there, this is Rachel from United Consumer Services. How are you today? I'm calling to inform you that our system has detected an overpayment on your … | {"risk": "high", "savings": "none"} |
| 1175 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, this is Officer Rachel Jenkins from the Social Security Administration. We've been trying to reach you, Mr. Johnson, about a serious issue with yo… | {"risk": "high", "suspiciousness": "very_suspicious"} |
| 1213 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, is this Karen Wilson? My name is David Lee and I'm calling from United Refund Services. We've been trying to reach you regarding a refund you're o… | {"risk": "high", "suspicion": "very_high"} |
| 1262 | True | False | False | missing_required_field | Innocent: Hello. Suspect: Hello, is this Mr. Harrison? My name is Officer Thompson from the Social Security Administration. I'm calling to inform you that we've received a notific… | {"risk": "high"} Explanation: The caller exhibits a scam pattern, a support scam variant, and displays a support refund scam. The caller appears to be a support refund scam operat… |
| 1320 | False | False | False | missing_required_field | Innocent: Hello. Suspect: Hi there! My name is Alex and I'm a licensed insurance representative. How are you today? Innocent: I'm doing great, Alex. Thanks for asking. What can yo… | {"risk": "high", "suspicion": "very_high"} |
| 1322 | True | False | False | malformed_json | Innocent: Hello. Suspect: Hello, is this Emily Wilson? I'm calling from the National Sweepstakes Commission. We've been trying to reach you about a prize you've won. You've been s… | {"risk": "high"} Explanation: The caller exhibits a scam pattern, a reward scam specifically. They display a reward scam pattern by offering a prize in exchange for a processing f… |

## Sample: baseline_wrong_llm_correct (11 total)

| index | gold | pred | valid | failure | transcript excerpt | raw output |
|---:|---|---|---|---|---|---|
| 33 | False | False | True | — | caller: Hello, this is John from the Social Security Administration. receiver: What can I do for you, Mr. John? caller: I'm calling about a routine matter regarding a tax refund r… | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate ssn call.", "flagged_spans": []} |
| 595 | False | False | True | — | caller: Hello, this is Karen Thompson from the Social Security Administration. I'm calling to discuss a security alert with your social security number, which indicates potential … | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate ssn call.", "flagged_spans": []} |
| 680 | True | True | True | — | caller: Hi, this is Emily from the Apple Refund Center. We've been trying to reach you about a refund you're owed for a purchase made through our website. receiver: Apple Refund C… | {"risk": "high", "fraud_type": "refund_scam", "reason": "Caller exhibits a refund scam pattern.", "flagged_spans": []} |
| 896 | False | False | True | — | caller: Hi, this is Sarah from Microsoft Support. We've received a report that your Microsoft account has been compromised. receiver: Oh no, really? What happened? caller: One of … | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate support call.", "flagged_spans": []} |
| 957 | False | False | True | — | caller: This is Robert from the electronics store. The TV you ordered has arrived. receiver: Perfect, can I schedule a delivery? caller: Certainly! What time suits you this week? | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 964 | True | True | True | — | caller: Good afternoon, am I speaking with Mr. Johnson? receiver: Yes, who is this? caller: I’m from the IRS. You owe back taxes and need to pay immediately to avoid arrest. | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 969 | True | True | True | — | Innocent: Hello. Suspect: Hi there, this is Rachel from Accounts Management at Greenfield Services. How are you today? I'm calling to inform you that our system has detected an ov… | {"risk": "high", "fraud_type": "refund_scam", "reason": "Caller exhibits a refund scam pattern.", "flagged_spans": []} |
| 982 | False | False | True | — | caller: Hello, this is Officer Johnson from the Social Security Administration. I'm calling to inform you that we've been trying to reach you regarding a matter with your social s… | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate ssn call.", "flagged_spans": []} |
| 1149 | False | False | True | — | caller: Hello, your meal delivery has arrived. Should I leave it at your door or hand it to you? receiver: Please hand it to me, I’m coming out now. | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 1157 | True | True | True | — | caller: Hello, I noticed you missed jury duty. receiver: I wasn’t notified about it. caller: Settle the fine by purchasing gift cards and reading the numbers. | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 1211 | True | True | True | — | caller: Hello, thank you for calling Google Chrome Support. We're experiencing a high volume of calls today, so I appreciate you waiting. receiver: Hi, I'm calling because I think… | {"risk": "high", "fraud_type": "tech_support_scam", "reason": "Caller exhibits a support scam pattern.", "flagged_spans": []} |
