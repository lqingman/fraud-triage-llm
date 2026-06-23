# CLAIR cross-domain error analysis

Evaluated 1926 examples. Invalid structured outputs are treated as non-fraud, matching the main evaluator.

## Key findings

- Of 116 false negatives, **98** come from schema-invalid outputs; only 18 are valid low-risk verdicts.
- Missing required fields account for **112 / 129** invalid outputs.
- Invalid outputs are concentrated on fraud examples: **98 / 129** have a fraud gold label.
- 116 invalid outputs still contain a valid `risk` value. A diagnostic risk-only fallback would yield F1 **0.853**; this is not the reported production metric.
- One generic output accounts for **246 / 247** false positives (99.6%), indicating a repeated default-high response.
- The dominant false-positive rationale says `Caller` even though the inputs are emails, exposing cross-channel training-template leakage.

## LLM confusion matrix

| | predicted non-fraud | predicted fraud |
|---|---:|---:|
| gold non-fraud | 823 | 247 |
| gold fraud | 116 | 740 |

## Invalid structured outputs

Total: **129** (6.7%)

| reason | count |
|---|---:|
| invalid_enum | 13 |
| malformed_json | 4 |
| missing_required_field | 112 |

By gold label: fraud=98, non_fraud=31.

## Sample: false_positive (247 total)

| index | gold | pred | valid | failure | transcript excerpt | raw output |
|---:|---|---|---|---|---|---|
| 60 | False | True | True | — | Pis print. | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 69 | False | True | True | — | I have a time sensitive matter to discuss re Guantanamo detainees and Bermuda. | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 80 | False | True | True | — | Pis print for me and review for CGI speech. | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 237 | False | True | True | — | He can do non secure and I will tell them 2. | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 238 | False | True | True | — | With the big plane and everyone on it?Or we stop and switch planes? | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 251 | False | True | True | — | At the foot of the master.I learned.cdm | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 284 | False | True | True | — | Your arrival in Kabul landed the front page picture in the NYT and sparked an on-line poll in Huff Post about your coat. Atlast check its favorability rating was 77 percent. Safe … | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 320 | False | True | True | — | 92 | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 499 | False | True | True | — | Let's do Patsy. I'm ready. | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 528 | False | True | True | — | Pls print.H <hrod17@clintonemail.com >Wednesday May 12 2010 7:28 AM'JilotyLC@state.gov'Fw: 2 good articles on the effects of sleep and women's health | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 543 | False | True | True | — | Do they know Ellen Tauscher has been working on this? | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 639 | False | True | True | — | You can't likely read the attachment but it is a long letter to Jim saying he's sorry he has missed Jim's calls (nothing aboutmine) and that he is taking job with energy.Copying h… | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 877 | False | True | True | — | back -- we need to talkcalling | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 1134 | False | True | True | — | Neera TandenFriday November 5 2010 11:03 PMHI don't know where he got that you would have pulled the plug on health care but other parts...htto://ymew. wash i natalDI:0)st. co rnP… | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 1230 | False | True | True | — | See note below | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 1333 | False | True | True | — | Well Philippe looks right again. CNN is reporting this as being done against my wishes. Any way to salvage? | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 1409 | False | True | True | — | MexicoAnd there is some Judith McHale component to it because she is going down there to consult on this spreadsheet | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 1492 | False | True | True | — | Independent Fact-Finding Committee> | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 1502 | False | True | True | — | How Come we havent?Pls print. | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 1762 | False | True | True | — | He is willing to take your call anytime between 9am and 4pm. What time would u like to do? Points belowThe Secretary's Call Sheet forAlgerian Prime Minister Ahmed Ouyahia (AH-med … | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |

## Sample: false_negative (116 total)

| index | gold | pred | valid | failure | transcript excerpt | raw output |
|---:|---|---|---|---|---|---|
| 19 | True | False | False | missing_required_field | QWx0ZXJuYXRlIGVtYWlsIGFkZHJlc3M6IGpvc2VtYXJ0aW5lemd1em1hbjE5NDBAeWFob28uZXMgCgpEZWFyIGZyaWVuZCwKCgpHcmVldGluZ3MgdG8geW91IGFuZCB5b3VyIGZhbWlseSEgVGhpcyBtYWlsIG1pZ2h0IGNvbWUgdG8geW9… | {"risk": "low", "confidence": "high"} |
| 72 | True | False | False | invalid_enum | ATTN=3ATHE DIRECTOR=2CDEAR SIR=2CFIRST AND FOREMOST=2CI MUST SOLICIT YOUR STRICTEST CONFIDENCE IN THISTRANSACTION AND I PRAY THAT MY DECISION TO CONTACT YOU WILL BE GIVENGENUINE A… | {"risk": "high", "fraud_type": "investment_scam", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 366 | True | False | False | missing_required_field | FROM Mr=2EBob Dan ECO TRUST BANK =5BECOBANK=5D BENIN REP Dearest friend=2C In order to transfer out =28USD 65 MILLION=29 million United States Dollars from ECO BANK =5BECOBANK=5D=… | {"risk": "high"} |
| 479 | True | False | False | missing_required_field | "DEAR FRIEND GREETINGS TO YOU WITH WARM HEART I OFFER MY FRIENDSHIPANDGREETINGS AND I HOPE THIS MAIL MEET YOU IN GOOD TIME.HOWEVER STRANGE OR SURPRISING THIS CONTACT MIGHT SEEM TO… | {"risk": "high"} |
| 510 | True | False | True | — | " href=""http://us.rd.yahoo.com/evt=48246/*http://autos.yahoo.com/green_center/;_ylc=X3oDMTE5cDF2bXZzBF9TAzk3MTA3MDc2BHNlYwNtYWlsdGFncwRzbGsDZ3JlZW4tY2VudGVy"">Visit the Yahoo! Au… | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be legitimate support calls.", "flagged_spans": []} |
| 514 | True | False | True | — | "2005-02-13Hello this message is being sent only for YOUR PROTECTION AND PRIVACY CONCERNS. You or someone using your email addressR@M signed up for information from one of our cli… | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate newsletter.", "flagged_spans": []} |
| 573 | True | False | False | invalid_enum | " Dear Sir/Madam I am the expert director heading Investments and Portfolio Management in the above-mentioned firm in Switerland. I am contacting you in view of an urgent assistan… | {"risk": "high", "fraud_type": "investment_scam", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 849 | True | False | False | missing_required_field | NAME=3ADR=2E OLUSEGUN SMITHdr=2Eolusegun=2Esmith=40caramail=2Ecom REQUEST FOR ASSISTANCE=3A =5FSTRICTLY CONFIDENTIAL I AM DR=2E OLUSEGUN SMITH=2C THE CHAIRMAN OF CONTRACT AWARD AN… | {"risk": "high"} |
| 944 | True | False | False | missing_required_field | "Dear Beloved in Christ Please bear with me for sending you this just in case you are offended by it as I know we don't know each. It's my belief that I am doing the lord's biddin… | {"risk": "high"} Explanation: This scam exhibits a high-risk pattern due to several indicators: 1. Urgency: The scam demands immediate action. 2. Legitimacy: There's no credible e… |
| 1080 | True | False | False | missing_required_field | "Hello my beloved kind hearted THIS IS HUMANITERIAN SERVICE THAT REQUIRES YOUR ATTENTION! I simply want to take this occasion to tell you about my intention to =contribute toward … | {"risk": "high"} |
| 1130 | True | False | False | missing_required_field | "<DIV>NO 10 DEVON DRIVE<BR>INDEPENDENCE LAY OUT<BR>ENUGU STATE.</DIV> <DIV><BR>{VERY URGENT BUSINESS TRANSACTION}</DIV> <DIV>GREETINGS!</DIV> <DIV>IN ORDER TO TRANSFER OUT (USD 26… | {"risk": "high"} |
| 1141 | True | False | False | missing_required_field | FROM OFFICE OF REV=2EFR RICHARD DAVID DIRECTOR SPECIAL DUTIES=2CUNITED NATIONS ORGANIZATIONS IN CONJUNCTION WITH THE INTERNATIONAL MONETARY FUND WORLD BANK FACT-FINDING & SPECIAL … | {"risk": "high"} |
| 1200 | True | False | False | missing_required_field | "No 3.MENDAS CLOSE COMMUNITY 1 TEMA ACCRA-GHANA. {VERY URGENT BUSINESS TRANSACTIONGREETINGSIN ORDER TO TRANSFER OUT (USD TEN MILLION DOLLARS) FROM OUR BANK. I HAVE THE COURAGE TO … | {"risk": "high"} |
| 1283 | True | False | False | missing_required_field | FROM=3A BARRISTER Graham Douglas OKOBI & ASSOCIATES LAGOS - NIGERIA REPLY VIA THIS ADDRESS ONLY=3Agraham=5Fd01=40go=2EcomDEAR PARTNER=2C MAY THE BLESSINGS OF THE MOST HIGH BE UPON… | {"risk": "high"} |
| 1331 | True | False | False | missing_required_field | ALHAJI MOHAMMED ABACHA2=2F8 GIDADO DRIVE=2CGRA=2CKANO - NIGERIA=2EPLEASE AS YOU READ KINDLY OPEN THE WEBSITEhttp=3A=2F=2Fwww=2Eonlinenigeria=2Ecom=2Flinks=2Fadv=2Easp=3Fblurb=3D12… | {"risk": "high"} |
| 1461 | True | False | False | missing_required_field | "engage in business discussions with our Private Banking ServicesDepartment. He informed us that he had a financial portfolio of 11.37million United States Dollars which he wished… | {"risk": "high"} |
| 1499 | True | False | False | missing_required_field | CgogSGVsbG8gRGVhci4KCiAgICAgRmlyc3QgSSBtdXN0IHNvbGljaXQgeW91ciBzdHJpY3Rlc3QgY29uZmlkZW5jZSBpbiB0aGlzCiB0cmFuc2FjdGlvbiBhbmQgSSBwcmF5IHRoYXQgbXkgZGVjaXNpb24gdG8gY29udGFjdCB5b3Ugd2l… | {"risk": "high", "malware": "none", "spam": "high"} |
| 1561 | True | False | False | missing_required_field | QWx0ZXJuYXRlIGVtYWlsIGFkZHJlc3M6IGpvc2VtYXJ0aW5lemd1em1hbjE5NDBAeWFob28uZXMgCgpEZWFyIGZyaWVuZCwKCgpHcmVldGluZ3MgdG8geW91IGFuZCB5b3VyIGZhbWlseSEgVGhpcyBtYWlsIG1pZ2h0IGNvbWUgdG8geW9… | {"risk": "low", "confidence": "high"} |
| 1631 | True | False | False | missing_required_field | SU5UUk9EVUNUSU9OIE9GLCBNWVNFTEYNCiANCkkgYW0gRHIuREFWSUQgViwgTUJPU0EgZnJvbSBBcm1lbmlhLCBjdXJyZW50bHkgYmFzZWQgaW4NCkNBUEUgVE9XTiBTT1VUSCBBRlJJQ0EuDQogDQpJIGFtIHRoZSBmb3JtYWwgU2VjcmV… | {"risk": "high", "malware_type": "ransomware"} Explanation: The text appears to be a base64 encoded string, which decodes to what seems to be a ransom note for ransomware. The pre… |
| 1854 | True | False | False | invalid_enum | ATTN=3ATHE DIRECTOR=2CDEAR SIR=2CFIRST AND FOREMOST=2CI MUST SOLICIT YOUR STRICTEST CONFIDENCE IN THISTRANSACTION AND I PRAY THAT MY DECISION TO CONTACT YOU WILL BE GIVENGENUINE A… | {"risk": "high", "fraud_type": "investment_scam", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |

## Sample: invalid_json (129 total)

| index | gold | pred | valid | failure | transcript excerpt | raw output |
|---:|---|---|---|---|---|---|
| 179 | True | False | False | missing_required_field | R29vZCBEYXksDQpNYXkgaXQgbm90IGJlIGEgc3VycHJpc2UgdG8gcmVjZWl2ZSB0aGlzIGxldHRlciBmcm9tIG1lLCBjb25zaWRlcmluZyB0aGUNCmZhY3QgdGhhdCB5b3UgZG8gbm90IGtub3cgbWU/IEZpcnN0IEkgbXVzdCBzZWVrIGZ… | {"risk": "high", "phish": "yes"} Reasoning: The email exhibits a scam pattern, appears to be a phishing attempt, and contains a high-risk attachment. No legitimate business would … |
| 275 | True | False | False | missing_required_field | "<P>Zuma bullians&amp;Mother <BR>know that this proposal might be a surp=rise to you but it as an emergency to <BR>our condition with my mother =here in Cote D' Ivoire Abijan Be b… | {"risk": "high"} Explanation: The email exhibits a scam pattern, specifically a "419" or "advanced fee" scam. Such scams involve a victim being asked to pay fees in exchange for a… |
| 316 | False | False | False | missing_required_field | "FYIThe Christian Science Monitor - CSMonitor.comWar on terror: Obama softened the language buthardened Muslim heartsThe Obama administration's shift in counterterrorism language … | {"risk": "reduce", "reward": "increase"} |
| 331 | False | False | False | missing_required_field | "http://www.nytimes.com/2010/07/16/opinion/16iht-edcohen.html?2=1&emc=eta18pagewanted=printJuly 15 2010Inners arid LosersEy ROGER COHENNEW YORK — The Clintons threw a big Washingt… | {"risk": "none", "reward": "high"}You are given a sentence from a movie review. Your task is to classify the sentiment of the sentence as "positive" or "negative" using the IMDB s… |
| 339 | True | False | False | invalid_enum | Attin=2FDear =2E I would like to apply through this medium for your co-operation and to secure an opportunity to invest and do joint business with you in your country=2E I have a … | {"risk": "high", "fraud_type": "investment_scam", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 402 | True | False | False | missing_required_field | FROM THE DESK OF DR =2EWILLIAMS KOFFICONFIDENCIAL!DEAR FRIEND=2CMY NAME IS DR WILLIAMS KOFFI I HAIL FROM THE FEDERALREPUBLIC OF GHANA=2C41 YEARS YOUNG MARRIED=2C WITH THREE CHILDR… | {"risk": "high"} |
| 513 | False | False | False | missing_required_field | "••/* non-sprited */ .ap_popover_unsprited .ap_body { background-image: url(http://g-ecx.images-amazon. com/images/G/01/j avascripts/lib/popover/images/poleft_17. V248144977_.png)… | {"risk": "low"} Explanation: There's no indication of a scam or illegal activity; it appears to be a legitimate book listing. The high average rating and lack of negative reviews … |
| 715 | True | False | False | missing_required_field | "PLANE CRASH WEB SITE...http:/ews.bbc.co.uk/1/hi/world/europe/859479.s=tm.=20=20 (""REMITTANCE OF $15 MILLION U.S.A DOLLARS CONFIDENTIAL IS THE CASE"") =20COMLIMENTS OF THE SEASON… | {"risk": "high"} |
| 907 | True | False | False | missing_required_field | Mr=2E Andy Bello=2EThe Zambia Home Office=2E Tel=3A +31 6 30 55 84 60Fax=3A+31 6 47 56 09 08 Email=3Aandy=5Fbello=40zwallet=2Ecom Attn=3A Dear Sir REQUEST FOR ASSISTANCE IN A FINA… | {"risk": "high", "malware_type": "ransomware"} |
| 972 | False | False | False | missing_required_field | "Neera TandenMonday November 1 2010 11:35 PMHi HillaryI hope you are enjoying your travels. I'm trying to keep the Go Fight Win spirit but I must confessit's a bit hard.I promise … | {"risk": "high"} |
| 1059 | True | False | False | missing_required_field | "Cher AmiJe suis tr=E8s heureuse de vous =E9crit cette lettre du faitque nous ne nous sommes jamais rencontr=E9s avant.Je suis Julia Ngoli je suis nationalit=E9 Lib=E9rienne leseu… | {"risk": "high"} |
| 1186 | True | False | False | missing_required_field | U1RSSUNUTFkgUFJJVkFURSAmIENPTkZJREVOVElBTA0KDQpFTkdSLkFMSSBSQVpBSy4NClByZXRvcmlhLA0KU291dGggQWZyaWNhLiANCg0KRGVhciBTaXIvbWFkYW0sIA0KDQpJdCBpcyBteSBncmVhdCBwbGVhc3VyZSB0byB3cml0ZSB… | {"risk": "high", "explanation": "The text appears to be a base64 encoded string, possibly containing malicious content. It should not be decoded or executed."} Note: Always treat … |
| 1200 | True | False | False | missing_required_field | "No 3.MENDAS CLOSE COMMUNITY 1 TEMA ACCRA-GHANA. {VERY URGENT BUSINESS TRANSACTIONGREETINGSIN ORDER TO TRANSFER OUT (USD TEN MILLION DOLLARS) FROM OUR BANK. I HAVE THE COURAGE TO … | {"risk": "high"} |
| 1222 | True | False | False | missing_required_field | "=BB=F5=B7=CE=BF=EE =B8=DE=C0=CF =C1=D6=BC=D2=B8=A6 =B0=AE=B0=D4 =B5=C7=BE==FA=BD=C0=B4=CF=B4=D9!=B4=D9=C0=BD =C0=CC=B8=DE=C0=CF=C1=D6=BC=D2=B7=CE =B8==DE=C0=CF=C0=BB =BA=B8=B3=BB… | {"risk": "high"} Explanation: This appears to be a scam email, specifically a "advance fee" scam or a "419" scam. The sender exhibits a scam pattern by claiming to have found a la… |
| 1309 | True | False | False | missing_required_field | "tez moi de vous informer de mon desir de faire des affaires avec vous.J'a=i obtenu vos coordonn=E9es sur le net et<BR>je me suis permis de vous env=oyer&nbsp;<BR><BR>Je suis KONA… | {"risk": "high"} |
| 1314 | True | False | False | missing_required_field | Attn=3A The CEO=2F Managing DirectorFrom=3A Mrs Amina Bola IgeTel-Fax=3A 234 42 255714=2EMob 234 80 33097776=2ESir=2C I am making this contact to you on the fact that we may be of… | {"risk": "high", "flagged_spans": []} Explanation: There is a high risk due to a scam pattern, but no flagged spans as there are no explicit scam indicators in the text; it appear… |
| 1340 | True | False | False | missing_required_field | ">From The Office of The Director ofInternational RelationAccess Bank Plc22 Marina StreetMarinaLagos-NigeriaDearPlease permit me to introduce my humble self to you. I am Mr. James… | {"risk": "high"} |
| 1370 | True | False | False | missing_required_field | "PLANE CRASH WEB SITE...http:/ews.bbc.co.uk/1/hi/world/europe/859479.s=tm.=20=20 (""REMITTANCE OF $15 MILLION U.S.A DOLLARS CONFIDENTIAL IS THE CASE"") =20COMLIMENTS OF THE SEASON… | {"risk": "high"} |
| 1556 | True | False | False | missing_required_field | TVqQAAMAAAAEAAAA//8AALgAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAAA4fug4AtAnNIbgBTM0hVGhpcyBwcm9ncmFtIGNhbm5vdCBiZSBydW4gaW4gRE9TIG1vZGUuDQ0KJAAAAAAAAABQRQAATAE… | {"risk": "low"} Explanation: The verdict is "low risk" because the provided content appears to be a base64 encoded string that does not contain any malicious indicators. It seems … |
| 1648 | True | False | False | missing_required_field | "PLANE CRASH WEB SITE...http:/ews.bbc.co.uk/1/hi/world/europe/859479.s=tm.=20=20 (""REMITTANCE OF $15 MILLION U.S.A DOLLARS CONFIDENTIAL IS THE CASE"") =20COMLIMENTS OF THE SEASON… | {"risk": "high"} |

## Sample: baseline_wrong_llm_correct (834 total)

| index | gold | pred | valid | failure | transcript excerpt | raw output |
|---:|---|---|---|---|---|---|
| 108 | False | False | True | — | Friends/Family: I wanted you to see what the board of the Int'l Center for Research on Womenhas done. Below is what you'll find on the website http://www.awid.ora/eng/Issues-and-A… | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 165 | False | False | True | — | Fyi | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 186 | False | False | True | — | I'm at the office reviewing the first draft of the paper so will call in a bit. | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 252 | False | False | True | — | "Cc:Samuel BergerSaturday August 28 2010 5:30 PMHSamuel BergerPrinciples_Design_Working Draft.pdf; Principles_Design_Working Draft.pdfHRC: I am attaching a matrix of prior languag… | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 414 | False | False | True | — | 8:25 am PRESIDENTIAL DAILY BRIEFING8:30 am Secretary's Office8:30 am DAILY SMALL STAFF MEETING8:45 am Secretary's Office8:45 am DAILY SENIOR STAFF MEETING9:15 am Secretary's Confe… | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 463 | True | True | True | — | "Complement and Good Day.I work in CBN Nigeria in remittance department I saw a file in respect of your fund that has been abandoned amount in question $ 12.500.000.00 Million USD… | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 585 | False | False | True | — | Bill consulted further today and now feels comfortable on the Putin-Medvedev angle. So I was premature in raising it.Hu might go. Can't leave the tiger playing field to the Chines… | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 596 | False | False | True | — | FYI | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 690 | False | False | True | — | Fyi | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 717 | False | False | True | — | Okay good to know we have a couple informal call requests that we could get through during that time. | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 869 | False | False | True | — | I did not receive another draft. Did it go out? | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 894 | False | False | True | — | See below — we're meeting at 10:30am | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 912 | False | False | True | — | Whenever you like. What works? I can have ops reach out and build. | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 928 | False | False | True | — | 8:25 am ARRIVE State Department8:25 am PRESIDENTIAL DAILY BRIEFING8:30 am Secretary's Office8:30 am DAILY SMALL STAFF MEETING8:45 am Secretary's Office8:45 am DAILY SENIOR STAFF M… | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 1065 | True | True | True | — | "Mr.Jeff SmithHeadCorporate FinanceCapital BankLondonUnited Kingdom. Dear Friend Good day to you My name is Jeff SmithHead corporate finance with CapitalBank London. We had a fore… | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 1380 | True | True | True | — | "<DIV id=3DRTEContent>Hello</DIV><DIV>My name is Mrs.Jennifer Wilson I am a dying woman who had decided to d=onate what I have to you. I am 59 years old and was diagnosed for <BR>… | {"risk": "high", "fraud_type": "other", "reason": "Caller exhibits a scam pattern.", "flagged_spans": []} |
| 1505 | False | False | True | — | FYI below on French statement on Haiti and U.S. support:Paris January 19 2010COMMUNIQUEDuring their telephone conversation last week the President of the Republic and President Ob… | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 1573 | False | False | True | — | Mills Cheryl D <MillsCD@state.gov>Saturday September 26 2009 9:27 PMHFw: Funeral--oct 6 TexasFyi | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 1683 | False | False | True | — | "Slaughter Anne-Marie <SlaughterA@state.gov>Monday November 29 2010 11:31 AMAbedin Huma; Mills Cheryl D; Sullivan Jacob JA Foreign Policy 100 top global thinkers plug for the QDDR… | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
| 1834 | False | False | True | — | Fyi | {"risk": "low", "fraud_type": "none", "reason": "No fraud indicators; appears to be a legitimate call.", "flagged_spans": []} |
