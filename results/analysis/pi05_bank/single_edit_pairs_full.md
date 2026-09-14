# Single-edit pairs, complete listing (426 pairs, 68 tasks)

Every pair of phrases on one LIBERO task that differ in exactly one contiguous token span, both measured on the same 50 initial states (n=50 each). Success in %, delta = B - A in pp; p = two-proportion z (Fisher exact and paired McNemar in the CSV). significant = |delta| >= 18 pp and p < 0.05 (115 of 426); the rest are nulls and are listed on purpose. sealed = task is in the sealed set v2 (seed 20260914); those pairs were removed from the distillation evidence but belong in the paper analysis. Source: results/analysis/pi05_bank/single_edit_pairs_full.csv.

| edit class | pairs | significant | mean abs delta |
|---|---|---|---|
| case | 4 | 0 | 1.5 |
| color | 18 | 4 | 8.3 |
| determiner | 2 | 0 | 2.0 |
| noun/other | 235 | 99 | 26.3 |
| preposition/particle | 18 | 1 | 4.1 |
| punctuation | 4 | 0 | 1.0 |
| verb/frame | 125 | 10 | 6.3 |
| words added | 11 | 1 | 6.9 |
| words removed | 9 | 0 | 0.4 |

| task | sealed | class | A | A % | B | B % | delta | p |
|---|---|---|---|---|---|---|---|---|
| libero_10/0 |  | words removed | put both the alphabet soup and the tomato sauce in the basket | 94 | put both the soup and the tomato sauce in the basket | 96 | +2 | 0.6464 |
| libero_10/1 |  | words removed | put both the cream cheese box and the butter in the basket | 100 | put both the cream cheese and the butter in the basket | 100 | +0 | 1.0000 |
| libero_10/2 |  | verb/frame | turn on the stove and put the moka pot on it | 94 | switch on the stove and put the moka pot on it | 94 | +0 | 1.0000 |
| libero_10/2 |  | noun/other | turn on the stove and put the moka pot on it | 94 | turn on the stove and put the coffee pot on it | 100 | +6 | 0.0786 |
| libero_10/3 |  | verb/frame | put the black bowl in the bottom drawer of the cabinet and close it | 98 | put the black bowl in the bottom drawer of the cabinet and shut it | 96 | -2 | 0.5577 |
| libero_10/5 |  | noun/other | pick up the book and place it in the back compartment of the caddy | 98 | pick up the book and place it in the back slot of the caddy | 100 | +2 | 0.3149 |
| libero_10/8 |  | noun/other | put both coffee pots on the stove | 66 | put the moka pots on the stove | 66 | +0 | 1.0000 |
| libero_10/8 |  | noun/other | put both moka pots on the stove | 62 | put both coffee pots on the stove | 66 | +4 | 0.6769 |
| libero_10/8 |  | noun/other | put both moka pots on the stove | 62 | put the moka pots on the stove | 66 | +4 | 0.6769 |
| libero_10/9 |  | noun/other | put the yellow and white mug in the microwave and close it | 94 | put the yellow and white cup in the microwave and close it | 94 | +0 | 1.0000 |
| libero_90/1 |  | verb/frame | close the top drawer of the cabinet and put the black bowl on top of it | 0 | close the top drawer of the cabinet and place the black bowl on top of it | 0 | +0 | 1.0000 |
| libero_90/1 |  | verb/frame | close the top drawer of the cabinet and put the black bowl on top of it | 0 | shut the top drawer of the cabinet and put the black bowl on top of it | 0 | +0 | 1.0000 |
| libero_90/10 |  | preposition/particle | place the black bowl on top of the cabinet | 100 | place the black bowl up onto the cabinet | 60 | **-40** | <0.0001 |
| libero_90/10 |  | color | put the black bowl on top of the cabinet | 94 | put the grey bowl on top of the cabinet | 96 | +2 | 0.6464 |
| libero_90/10 |  | verb/frame | put the black bowl on top of the cabinet | 94 | place the black bowl on top of the cabinet | 100 | +6 | 0.0786 |
| libero_90/15 |  | preposition/particle | put the middle black bowl on top of the cabinet | 2 | put the middle black bowl atop the cabinet | 0 | -2 | 0.3149 |
| libero_90/15 |  | verb/frame | put the middle black bowl on top of the cabinet | 2 | place the middle black bowl on top of the cabinet | 10 | +8 | 0.0921 |
| libero_90/17 |  | verb/frame | stack the middle black bowl on the back black bowl | 0 | put the middle black bowl on the back black bowl | 0 | +0 | 1.0000 |
| libero_90/17 |  | noun/other | stack the middle black bowl on the back black bowl | 0 | stack the centre black bowl on the back black bowl | 0 | +0 | 1.0000 |
| libero_90/17 |  | noun/other | stack the middle black bowl on the back black bowl | 0 | stack the middle black bowl on the rear black bowl | 0 | +0 | 1.0000 |
| libero_90/19 |  | noun/other | put the moka pot on the stove | 94 | put the coffee pot on the stove | 84 | -10 | 0.1100 |
| libero_90/19 |  | noun/other | put the moka pot on the stove | 94 | put the espresso pot on the stove | 84 | -10 | 0.1100 |
| libero_90/19 |  | noun/other | put the moka pot on the burner | 96 | put the moka pot on the stove. | 90 | -6 | 0.2397 |
| libero_90/19 |  | noun/other | put the moka pot on the burner | 96 | put the moka pot on the hot plate | 92 | -4 | 0.3997 |
| libero_90/19 |  | punctuation | put the moka pot on the stove | 94 | put the moka pot on the stove. | 90 | -4 | 0.4610 |
| libero_90/19 |  | case | put the moka pot on the stove | 94 | Put the moka pot on the stove | 92 | -2 | 0.6951 |
| libero_90/19 |  | verb/frame | can you put the moka pot on the stove | 94 | i want the moka pot on the stove | 92 | -2 | 0.6951 |
| libero_90/19 |  | noun/other | can you put the moka pot on the stove | 94 | just put the moka pot on the stove | 92 | -2 | 0.6951 |
| libero_90/19 |  | verb/frame | put the moka pot on the stove | 94 | i want the moka pot on the stove | 92 | -2 | 0.6951 |
| libero_90/19 |  | words added | put the moka pot on the stove | 94 | just put the moka pot on the stove | 92 | -2 | 0.6951 |
| libero_90/19 |  | noun/other | put the moka pot on the stove | 94 | put the moka pot on the hot plate | 92 | -2 | 0.6951 |
| libero_90/19 |  | noun/other | put the moka pot on the hot plate | 92 | put the moka pot on the stove. | 90 | -2 | 0.7268 |
| libero_90/19 |  | words added | put the moka pot on the stove | 94 | put the silver moka pot on the stove | 92 | -2 | 0.6951 |
| libero_90/19 |  | verb/frame | Put the moka pot on the stove | 92 | i want the moka pot on the stove | 92 | +0 | 1.0000 |
| libero_90/19 |  | words added | Put the moka pot on the stove | 92 | just put the moka pot on the stove | 92 | +0 | 1.0000 |
| libero_90/19 |  | verb/frame | put the moka pot on the stove | 94 | can you put the moka pot on the stove | 94 | +0 | 1.0000 |
| libero_90/19 |  | noun/other | i want the moka pot on the stove | 92 | just put the moka pot on the stove | 92 | +0 | 1.0000 |
| libero_90/19 |  | noun/other | put the coffee pot on the stove | 84 | put the espresso pot on the stove | 84 | +0 | 1.0000 |
| libero_90/19 |  | verb/frame | the moka pot goes on the stove | 90 | the moka pot should be put on the stove | 90 | +0 | 1.0000 |
| libero_90/19 |  | verb/frame | Put the moka pot on the stove | 92 | can you put the moka pot on the stove | 94 | +2 | 0.6951 |
| libero_90/19 |  | noun/other | put the moka pot on the stove | 94 | put the moka pot on the burner | 96 | +2 | 0.6464 |
| libero_90/19 |  | noun/other | put the coffee pot on the stove | 84 | put the silver moka pot on the stove | 92 | +8 | 0.2184 |
| libero_90/19 |  | noun/other | put the espresso pot on the stove | 84 | put the silver moka pot on the stove | 92 | +8 | 0.2184 |
| libero_90/21 |  | verb/frame | fire up the stove and put the frying pan on it | 0 | start the stove and put the frying pan on it | 0 | +0 | 1.0000 |
| libero_90/21 |  | verb/frame | fire up the stove and put the frying pan on it | 0 | switch on the stove and put the frying pan on it | 0 | +0 | 1.0000 |
| libero_90/21 |  | verb/frame | turn on the stove and put the frying pan on it | 0 | fire up the stove and put the frying pan on it | 0 | +0 | 1.0000 |
| libero_90/21 |  | verb/frame | start the stove and put the frying pan on it | 0 | switch on the stove and put the frying pan on it | 0 | +0 | 1.0000 |
| libero_90/21 |  | verb/frame | turn on the stove and put the frying pan on it | 0 | start the stove and put the frying pan on it | 0 | +0 | 1.0000 |
| libero_90/21 |  | verb/frame | turn on the stove and put the frying pan on it | 0 | switch on the stove and put the frying pan on it | 0 | +0 | 1.0000 |
| libero_90/22 |  | noun/other | close the bottom drawer of the cupboard | 100 | close the bottom drawer of the wooden cabinet | 98 | -2 | 0.3149 |
| libero_90/22 |  | color | close the bottom drawer of the cabinet | 96 | close the bottom drawer of the wooden cabinet | 98 | +2 | 0.5577 |
| libero_90/22 |  | noun/other | close the bottom drawer of the cabinet | 96 | close the bottom drawer of the cupboard | 100 | +4 | 0.1531 |
| libero_90/22 |  | noun/other | close the bottom drawer of the cabinet | 96 | close the lower drawer of the cabinet | 100 | +4 | 0.1531 |
| libero_90/23 |  | verb/frame | close the bottom drawer of the cabinet and open the top drawer | 0 | push in the bottom drawer of the cabinet and open the top drawer | 0 | +0 | 1.0000 |
| libero_90/23 |  | verb/frame | close the bottom drawer of the cabinet and open the top drawer | 0 | shut the bottom drawer of the cabinet and open the top drawer | 0 | +0 | 1.0000 |
| libero_90/23 |  | verb/frame | push in the bottom drawer of the cabinet and open the top drawer | 0 | shut the bottom drawer of the cabinet and open the top drawer | 0 | +0 | 1.0000 |
| libero_90/24 |  | verb/frame | can you put the black bowl in the bottom drawer of the cabinet | 100 | i want the black bowl in the bottom drawer of the cabinet | 100 | +0 | 1.0000 |
| libero_90/24 |  | verb/frame | can you put the black bowl in the bottom drawer of the cabinet | 100 | place the black bowl in the bottom drawer of the cabinet | 100 | +0 | 1.0000 |
| libero_90/24 |  | verb/frame | put the black bowl in the bottom drawer of the cabinet | 100 | can you put the black bowl in the bottom drawer of the cabinet | 100 | +0 | 1.0000 |
| libero_90/24 |  | verb/frame | i want the black bowl in the bottom drawer of the cabinet | 100 | place the black bowl in the bottom drawer of the cabinet | 100 | +0 | 1.0000 |
| libero_90/24 |  | verb/frame | put the black bowl in the bottom drawer of the cabinet | 100 | i want the black bowl in the bottom drawer of the cabinet | 100 | +0 | 1.0000 |
| libero_90/24 |  | verb/frame | put the black bowl in the bottom drawer of the cabinet | 100 | place the black bowl in the bottom drawer of the cabinet | 100 | +0 | 1.0000 |
| libero_90/24 |  | noun/other | put the black bowl in the bottom drawer of the cabinet | 100 | put the black bowl in the bottom drawer of the cupboard | 100 | +0 | 1.0000 |
| libero_90/24 |  | color | put the black bowl in the bottom drawer of the cabinet | 100 | put the black bowl in the bottom drawer of the wooden cabinet | 100 | +0 | 1.0000 |
| libero_90/24 |  | noun/other | put the black bowl in the bottom drawer of the cabinet | 100 | put the black dish in the bottom drawer of the cabinet | 100 | +0 | 1.0000 |
| libero_90/24 |  | noun/other | put the black bowl in the bottom drawer of the cupboard | 100 | put the black bowl in the bottom drawer of the wooden cabinet | 100 | +0 | 1.0000 |
| libero_90/24 |  | verb/frame | the black bowl goes in the bottom drawer of the cabinet | 100 | the black bowl should be put in the bottom drawer of the cabinet | 100 | +0 | 1.0000 |
| libero_90/25 |  | verb/frame | put the black bowl on top of the cabinet | 0 | place the black bowl on top of the cabinet | 0 | +0 | 1.0000 |
| libero_90/25 |  | preposition/particle | put the black bowl on top of the cabinet | 0 | put the black bowl atop the cabinet | 0 | +0 | 1.0000 |
| libero_90/25 |  | color | put the black bowl on top of the cabinet | 0 | put the bowl on top of the cabinet | 0 | +0 | 1.0000 |
| libero_90/25 |  | color | put the black bowl on top of the cabinet | 0 | put the grey bowl on top of the cabinet | 0 | +0 | 1.0000 |
| libero_90/25 |  | color | put the bowl on top of the cabinet | 0 | put the grey bowl on top of the cabinet | 0 | +0 | 1.0000 |
| libero_90/28 |  | verb/frame | close the open top drawer of the white cabinet on the left | 48 | push in the open top drawer of the white cabinet on the left | 30 | -18 | 0.0650 |
| libero_90/28 |  | noun/other | close the top drawer of the cabinet | 38 | close the upper drawer of the cabinet | 38 | +0 | 1.0000 |
| libero_90/30 |  | noun/other | put the black bowl on the plate | 76 | put the black bowl on the dish | 20 | **-56** | <0.0001 |
| libero_90/30 |  | noun/other | put the black bowl on the plate | 76 | put the dark bowl on the plate | 72 | -4 | 0.6484 |
| libero_90/30 |  | color | put the black bowl on the plate | 76 | put the grey bowl on the plate | 72 | -4 | 0.6484 |
| libero_90/30 |  | noun/other | put the dark bowl on the plate | 72 | put the grey bowl on the plate | 72 | +0 | 1.0000 |
| libero_90/31 |  | color | put the bowl on top of the cabinet | 36 | put the grey bowl on top of the cabinet | 10 | **-26** | 0.0020 |
| libero_90/31 |  | preposition/particle | put the black bowl on top of the cabinet | 16 | put the black bowl atop the cabinet | 10 | -6 | 0.3724 |
| libero_90/31 |  | color | put the black bowl on top of the cabinet | 16 | put the grey bowl on top of the cabinet | 10 | -6 | 0.3724 |
| libero_90/31 |  | verb/frame | put the black bowl on top of the cabinet | 16 | place the black bowl on top of the cabinet | 14 | -2 | 0.7794 |
| libero_90/31 |  | color | put the black bowl on top of the cabinet | 16 | put the bowl on top of the cabinet | 36 | **+20** | 0.0226 |
| libero_90/32 |  | verb/frame | put the ketchup in the top drawer of the cabinet | 0 | place the ketchup in the top drawer of the cabinet | 0 | +0 | 1.0000 |
| libero_90/32 |  | verb/frame | place the ketchup in the top drawer of the cabinet | 0 | set the ketchup in the top drawer of the cabinet | 0 | +0 | 1.0000 |
| libero_90/32 |  | preposition/particle | put the ketchup in the top drawer of the cabinet | 0 | put the ketchup into the top drawer of the cabinet | 0 | +0 | 1.0000 |
| libero_90/32 |  | verb/frame | put the ketchup in the top drawer of the cabinet | 0 | set the ketchup in the top drawer of the cabinet | 0 | +0 | 1.0000 |
| libero_90/34 |  | verb/frame | put the yellow and white mug to the front of the white mug | 0 | move the yellow and white mug to the front of the white mug | 0 | +0 | 1.0000 |
| libero_90/34 |  | verb/frame | place the yellow and white mug in front of the white mug | 0 | put the yellow and white mug in front of the white mug | 0 | +0 | 1.0000 |
| libero_90/34 |  | preposition/particle | put the yellow and white mug to the front of the white mug | 0 | put the yellow and white mug in front of the white mug | 0 | +0 | 1.0000 |
| libero_90/35 | yes | words added | open the microwave | 8 | open the microwave door | 2 | -6 | 0.1687 |
| libero_90/35 | yes | verb/frame | open the microwave | 8 | pull open the microwave | 6 | -2 | 0.6951 |
| libero_90/35 | yes | noun/other | open the microwave | 8 | unlatch the microwave | 6 | -2 | 0.6951 |
| libero_90/35 | yes | noun/other | open up the microwave | 8 | unlatch the microwave | 6 | -2 | 0.6951 |
| libero_90/35 | yes | preposition/particle | open the microwave | 8 | open up the microwave | 8 | +0 | 1.0000 |
| libero_90/35 | yes | noun/other | pull open the microwave | 6 | unlatch the microwave | 6 | +0 | 1.0000 |
| libero_90/36 |  | verb/frame | place the white bowl on the plate | 4 | set the white bowl on the plate | 2 | -2 | 0.5577 |
| libero_90/36 |  | noun/other | put the white bowl on the plate | 0 | put the white dish on the plate | 2 | +2 | 0.3149 |
| libero_90/36 |  | verb/frame | put the white bowl on the plate | 0 | set the white bowl on the plate | 2 | +2 | 0.3149 |
| libero_90/36 |  | verb/frame | put the white bowl on the plate | 0 | place the white bowl on the plate | 4 | +4 | 0.1531 |
| libero_90/37 |  | verb/frame | put the white bowl to the right of the plate | 0 | place the white bowl to the right of the plate | 0 | +0 | 1.0000 |
| libero_90/37 |  | preposition/particle | put the white bowl to the right of the plate | 0 | put the white bowl on the right of the plate | 0 | +0 | 1.0000 |
| libero_90/38 |  | noun/other | put the right moka pot on the stove | 36 | put the right moka pot on the hot plate | 40 | +4 | 0.6803 |
| libero_90/38 |  | verb/frame | lift the right moka pot and place it on the stove | 34 | pick up the right moka pot and place it on the stove | 54 | **+20** | 0.0440 |
| libero_90/39 |  | verb/frame | turn off the stove | 2 | power off the stove | 0 | -2 | 0.3149 |
| libero_90/39 |  | verb/frame | turn off the stove | 2 | shut off the stove | 0 | -2 | 0.3149 |
| libero_90/39 |  | verb/frame | turn off the stove | 2 | switch off the stove | 0 | -2 | 0.3149 |
| libero_90/39 |  | verb/frame | power off the stove | 0 | shut off the stove | 0 | +0 | 1.0000 |
| libero_90/39 |  | verb/frame | power off the stove | 0 | switch off the stove | 0 | +0 | 1.0000 |
| libero_90/39 |  | verb/frame | shut off the stove | 0 | switch off the stove | 0 | +0 | 1.0000 |
| libero_90/41 |  | preposition/particle | put the frying pan on top of the cabinet | 0 | put the frying pan atop the cabinet | 0 | +0 | 1.0000 |
| libero_90/41 |  | verb/frame | put the frying pan on top of the cabinet | 0 | place the frying pan on top of the cabinet | 2 | +2 | 0.3149 |
| libero_90/44 |  | verb/frame | fire up the stove | 100 | ignite the stove | 100 | +0 | 1.0000 |
| libero_90/44 |  | verb/frame | fire up the stove | 100 | power on the stove | 100 | +0 | 1.0000 |
| libero_90/44 |  | verb/frame | fire up the stove | 100 | start the stove | 100 | +0 | 1.0000 |
| libero_90/44 |  | verb/frame | fire up the stove | 100 | switch on the stove | 100 | +0 | 1.0000 |
| libero_90/44 |  | verb/frame | ignite the stove | 100 | power on the stove | 100 | +0 | 1.0000 |
| libero_90/44 |  | verb/frame | ignite the stove | 100 | start the stove | 100 | +0 | 1.0000 |
| libero_90/44 |  | verb/frame | ignite the stove | 100 | switch on the stove | 100 | +0 | 1.0000 |
| libero_90/44 |  | verb/frame | power on the stove | 100 | start the stove | 100 | +0 | 1.0000 |
| libero_90/44 |  | verb/frame | power on the stove | 100 | switch on the stove | 100 | +0 | 1.0000 |
| libero_90/44 |  | verb/frame | start the stove | 100 | switch on the stove | 100 | +0 | 1.0000 |
| libero_90/44 |  | verb/frame | turn on the stove | 6 | fire up the stove | 100 | **+94** | <0.0001 |
| libero_90/44 |  | verb/frame | turn on the stove | 6 | ignite the stove | 100 | **+94** | <0.0001 |
| libero_90/44 |  | verb/frame | turn on the stove | 6 | power on the stove | 100 | **+94** | <0.0001 |
| libero_90/44 |  | verb/frame | turn on the stove | 6 | start the stove | 100 | **+94** | <0.0001 |
| libero_90/44 |  | verb/frame | turn on the stove | 6 | switch on the stove | 100 | **+94** | <0.0001 |
| libero_90/48 |  | verb/frame | pick up the ketchup and put it in the basket | 2 | pick up the ketchup and place it in the basket | 0 | -2 | 0.3149 |
| libero_90/48 |  | preposition/particle | pick up the ketchup and put it in the basket | 2 | pick up the ketchup and put it into the basket | 0 | -2 | 0.3149 |
| libero_90/48 |  | verb/frame | pick up the ketchup and put it in the basket | 2 | pick up the ketchup and set it in the basket | 0 | -2 | 0.3149 |
| libero_90/48 |  | verb/frame | pick up the ketchup and put it in the basket | 2 | grab the ketchup and put it in the basket | 2 | +0 | 1.0000 |
| libero_90/48 |  | verb/frame | pick up the ketchup and place it in the basket | 0 | pick up the ketchup and set it in the basket | 0 | +0 | 1.0000 |
| libero_90/51 |  | verb/frame | pick up the butter and put it in the basket | 100 | lift the butter and put it in the basket | 100 | +0 | 1.0000 |
| libero_90/51 |  | verb/frame | pick up the butter and put it in the basket | 100 | pick up the butter and place it in the basket | 100 | +0 | 1.0000 |
| libero_90/52 |  | verb/frame | pick up the milk and put it in the basket | 0 | grab the milk and put it in the basket | 0 | +0 | 1.0000 |
| libero_90/52 |  | verb/frame | pick up the milk and put it in the basket | 0 | pick up the milk and place it in the basket | 0 | +0 | 1.0000 |
| libero_90/52 |  | verb/frame | pick up the milk and place it in the basket | 0 | pick up the milk and set it in the basket | 0 | +0 | 1.0000 |
| libero_90/52 |  | preposition/particle | pick up the milk and put it in the basket | 0 | pick up the milk and put it into the basket | 0 | +0 | 1.0000 |
| libero_90/52 |  | verb/frame | pick up the milk and put it in the basket | 0 | pick up the milk and set it in the basket | 0 | +0 | 1.0000 |
| libero_90/57 |  | verb/frame | grab the cream cheese and put it in the tray | 100 | lift the cream cheese and put it in the tray | 100 | +0 | 1.0000 |
| libero_90/57 |  | verb/frame | pick up the cream cheese and put it in the tray | 98 | pick up the cream cheese and place it in the tray | 98 | +0 | 1.0000 |
| libero_90/57 |  | verb/frame | pick up the cream cheese and put it in the tray | 98 | grab the cream cheese and put it in the tray | 100 | +2 | 0.3149 |
| libero_90/57 |  | verb/frame | pick up the cream cheese and put it in the tray | 98 | lift the cream cheese and put it in the tray | 100 | +2 | 0.3149 |
| libero_90/57 |  | words added | pick up the cream cheese and put it in the tray | 98 | pick up the cream cheese box and put it in the tray | 100 | +2 | 0.3149 |
| libero_90/59 | yes | noun/other | grasp the green and orange can and put it in the wooden tray | 22 | grasp the green and orange tin and put it in the wooden tray | 22 | +0 | 1.0000 |
| libero_90/59 | yes | noun/other | grasp the tomato sauce can and put it in the wooden tray | 30 | pick up the tomato sauce can and put it in the wooden tray | 30 | +0 | 1.0000 |
| libero_90/59 | yes | color | pick up the tomato sauce can and put it in the tray | 24 | pick up the tomato sauce can and put it in the wooden tray | 30 | +6 | 0.4992 |
| libero_90/59 | yes | noun/other | grasp the green and orange can and put it in the wooden tray | 22 | grasp the tomato sauce can and put it in the wooden tray | 30 | +8 | 0.3618 |
| libero_90/59 | yes | noun/other | grasp the green and orange tin and put it in the wooden tray | 22 | grasp the tomato sauce can and put it in the wooden tray | 30 | +8 | 0.3618 |
| libero_90/59 | yes | noun/other | grasp the orange and green can and put it in the wooden tray | 18 | grasp the tomato sauce can and put it in the wooden tray | 30 | +12 | 0.1601 |
| libero_90/59 | yes | verb/frame | grasp the tomato sauce can and place it in the wooden tray | 18 | grasp the tomato sauce can and put it in the wooden tray | 30 | +12 | 0.1601 |
| libero_90/66 |  | verb/frame | put the red mug on the right plate | 0 | place the red mug on the right plate | 0 | +0 | 1.0000 |
| libero_90/66 |  | verb/frame | place the red mug on the right plate | 0 | set the red mug on the right plate | 0 | +0 | 1.0000 |
| libero_90/66 |  | noun/other | put the red mug on the right plate | 0 | put the red cup on the right plate | 0 | +0 | 1.0000 |
| libero_90/66 |  | verb/frame | put the red mug on the right plate | 0 | set the red mug on the right plate | 0 | +0 | 1.0000 |
| libero_90/68 | yes | verb/frame | can you put the yellow and white mug on the right plate | 100 | i want the yellow and white mug on the right plate | 100 | +0 | 1.0000 |
| libero_90/68 | yes | verb/frame | put the yellow and white mug on the right plate | 100 | can you put the yellow and white mug on the right plate | 100 | +0 | 1.0000 |
| libero_90/68 | yes | verb/frame | put the yellow and white mug on the right plate | 100 | i want the yellow and white mug on the right plate | 100 | +0 | 1.0000 |
| libero_90/68 | yes | noun/other | put the yellow and white mug on the right plate | 100 | put the yellow and white cup on the right plate | 100 | +0 | 1.0000 |
| libero_90/68 | yes | noun/other | put the yellow and white mug on the right plate | 100 | put the yellow and white mug on the right dish | 100 | +0 | 1.0000 |
| libero_90/68 | yes | noun/other | put the yellow and white mug on the right dish | 100 | put the yellow and white mug on the right white plate | 100 | +0 | 1.0000 |
| libero_90/68 | yes | color | put the yellow and white mug on the right plate | 100 | put the yellow and white mug on the right white plate | 100 | +0 | 1.0000 |
| libero_90/68 | yes | verb/frame | the yellow and white mug goes on the right plate | 100 | the yellow and white mug should be put on the right plate | 100 | +0 | 1.0000 |
| libero_90/70 | yes | noun/other | set the brown pudding down to the right of that plate | 76 | set the chocolate pudding down to the right of that plate | 70 | -6 | 0.4992 |
| libero_90/70 | yes | noun/other | put the chocolate pudding to the right of the plate | 64 | put the chocolate pudding to the right of the dish | 70 | +6 | 0.5235 |
| libero_90/70 | yes | noun/other | put the chocolate pudding to the right of the plate | 64 | put the chocolate dessert to the right of the plate | 80 | +16 | 0.0748 |
| libero_90/71 |  | verb/frame | put the red mug on the plate | 0 | place the red mug on the plate | 0 | +0 | 1.0000 |
| libero_90/71 |  | verb/frame | place the red mug on the plate | 0 | set the red mug on the plate | 0 | +0 | 1.0000 |
| libero_90/71 |  | noun/other | put the red mug on the plate | 0 | put the red cup on the plate | 0 | +0 | 1.0000 |
| libero_90/71 |  | verb/frame | put the red mug on the plate | 0 | set the red mug on the plate | 0 | +0 | 1.0000 |
| libero_90/72 |  | noun/other | put the white mug on the dish | 100 | put the white mug on the plate. | 94 | -6 | 0.0786 |
| libero_90/72 |  | noun/other | put the white mug on the dish | 100 | put the white mug on the round plate | 96 | -4 | 0.1531 |
| libero_90/72 |  | determiner | Put the white mug on the plate | 98 | put that white mug on the plate | 98 | +0 | 1.0000 |
| libero_90/72 |  | noun/other | put the white mug on the plate | 94 | put the white cup on the plate | 94 | +0 | 1.0000 |
| libero_90/72 |  | punctuation | put the white mug on the plate | 94 | put the white mug on the plate. | 94 | +0 | 1.0000 |
| libero_90/72 |  | words added | put the white mug on the plate | 94 | put the white mug on the round plate | 96 | +2 | 0.6464 |
| libero_90/72 |  | noun/other | put the white mug on the plate. | 94 | put the white mug on the round plate | 96 | +2 | 0.6464 |
| libero_90/72 |  | case | put the white mug on the plate | 94 | Put the white mug on the plate | 98 | +4 | 0.3074 |
| libero_90/72 |  | determiner | put the white mug on the plate | 94 | put that white mug on the plate | 98 | +4 | 0.3074 |
| libero_90/72 |  | noun/other | put the white mug on the plate | 94 | put the white mug on the dish | 100 | +6 | 0.0786 |
| libero_90/78 |  | noun/other | pick up the book and place it in the front compartment of the caddy | 0 | pick up the book and place it in the front slot of the caddy | 0 | +0 | 1.0000 |
| libero_90/78 |  | verb/frame | pick up the book and place it in the front compartment of the caddy | 0 | pick up the book and put it in the front compartment of the caddy | 0 | +0 | 1.0000 |
| libero_90/78 |  | verb/frame | place the book in the front compartment of the caddy | 0 | put the book in the front compartment of the caddy | 0 | +0 | 1.0000 |
| libero_90/81 |  | noun/other | pick up the book and place it in the front compartment of the caddy | 0 | pick up the book and place it in the front slot of the caddy | 0 | +0 | 1.0000 |
| libero_90/81 |  | verb/frame | pick up the book and place it in the front compartment of the caddy | 0 | pick up the book and put it in the front compartment of the caddy | 0 | +0 | 1.0000 |
| libero_90/81 |  | verb/frame | place the book in the front compartment of the caddy | 0 | put the book in the front compartment of the caddy | 0 | +0 | 1.0000 |
| libero_90/82 |  | preposition/particle | place the black book inside the left slot of the brown caddy | 38 | place the black book into the left slot of the brown caddy | 34 | -4 | 0.6769 |
| libero_90/82 |  | noun/other | place the black book into the left compartment of the brown caddy | 22 | place the black book into the left slot of the brown caddy | 34 | +12 | 0.1814 |
| libero_90/9 |  | verb/frame | put the black bowl on the plate | 98 | can you put the black bowl on the plate | 84 | -14 | 0.0144 |
| libero_90/9 |  | noun/other | put the black bowl on the plate | 98 | put the dark bowl on the plate | 88 | -10 | 0.0500 |
| libero_90/9 |  | noun/other | put the black bowl on the plate | 98 | put the black bowl on the dish | 90 | -8 | 0.0921 |
| libero_90/9 |  | preposition/particle | put the black bowl on the plate | 98 | put the black bowl onto the plate | 90 | -8 | 0.0921 |
| libero_90/9 |  | color | put the black bowl on the plate | 98 | put the grey bowl on the plate | 90 | -8 | 0.0921 |
| libero_90/9 |  | verb/frame | put the black bowl on the plate | 98 | i want the black bowl on the plate | 94 | -4 | 0.3074 |
| libero_90/9 |  | noun/other | put the dark bowl on the plate | 88 | put the grey bowl on the plate | 90 | +2 | 0.7493 |
| libero_90/9 |  | verb/frame | the black bowl goes on the plate | 92 | the black bowl should be put on the plate | 94 | +2 | 0.6951 |
| libero_90/9 |  | verb/frame | can you put the black bowl on the plate | 84 | i want the black bowl on the plate | 94 | +10 | 0.1100 |
| libero_goal/1 | yes | noun/other | place the grey bowl on the electric burner | 92 | place the grey bowl on the electric hot plate | 0 | **-92** | <0.0001 |
| libero_goal/1 | yes | noun/other | place the grey bowl on the electric burner | 92 | place the grey bowl on the hot plate | 0 | **-92** | <0.0001 |
| libero_goal/1 | yes | noun/other | place the grey bowl on the electric burner | 92 | place the grey bowl on the hotplate | 18 | **-74** | <0.0001 |
| libero_goal/1 | yes | words removed | place the grey bowl on the electric hot plate | 0 | place the grey bowl on the hot plate | 0 | +0 | 1.0000 |
| libero_goal/1 | yes | noun/other | place the grey bowl on the electric hot plate | 0 | place the grey bowl on the hotplate | 18 | **+18** | 0.0017 |
| libero_goal/1 | yes | noun/other | place the grey bowl on the hot plate | 0 | place the grey bowl on the hotplate | 18 | **+18** | 0.0017 |
| libero_goal/2 |  | noun/other | put the wine bottle on top of the cabinet | 100 | put the wine jug on top of the cabinet | 96 | -4 | 0.1531 |
| libero_goal/2 |  | noun/other | put the wine bottle on top of the cabinet | 100 | put the wine bottle on top of the cupboard | 98 | -2 | 0.3149 |
| libero_goal/2 |  | preposition/particle | put the wine bottle on top of the cabinet | 100 | put the wine bottle up onto the cabinet | 98 | -2 | 0.3149 |
| libero_goal/2 |  | noun/other | put the wine bottle up onto the cabinet | 98 | put the wine jug on top of the cabinet | 96 | -2 | 0.5577 |
| libero_goal/4 |  | noun/other | put the bowl on top of the cabinet | 98 | put the dish on top of the cabinet | 92 | -6 | 0.1687 |
| libero_goal/4 |  | preposition/particle | put the bowl on top of the cabinet | 98 | put the bowl up onto the cabinet | 94 | -4 | 0.3074 |
| libero_goal/4 |  | noun/other | put the bowl up onto the cabinet | 94 | put the dish on top of the cabinet | 92 | -2 | 0.6951 |
| libero_goal/4 |  | verb/frame | put the bowl on top of the cabinet | 98 | can you put the bowl on top of the cabinet | 98 | +0 | 1.0000 |
| libero_goal/4 |  | verb/frame | can you put the bowl on top of the cabinet | 98 | i want the bowl on top of the cabinet | 100 | +2 | 0.3149 |
| libero_goal/4 |  | verb/frame | put the bowl on top of the cabinet | 98 | i want the bowl on top of the cabinet | 100 | +2 | 0.3149 |
| libero_goal/4 |  | noun/other | put the bowl on top of the cabinet | 98 | put the bowl on top of the cupboard | 100 | +2 | 0.3149 |
| libero_goal/4 |  | verb/frame | the bowl goes on top of the cabinet | 96 | the bowl should be put on top of the cabinet | 100 | +4 | 0.1531 |
| libero_goal/5 |  | noun/other | Go ahead and push the plate right up to the front of the stove. | 98 | Slide that dish over to the front of the stove. | 0 | **-98** | <0.0001 |
| libero_goal/5 |  | noun/other | push the plate to the front of the stove | 98 | slide that flat dish over to the front of the stove | 0 | **-98** | <0.0001 |
| libero_goal/5 |  | noun/other | push the plate to the front of the stove | 98 | slide that dish over to the front of the stove | 2 | **-96** | <0.0001 |
| libero_goal/5 |  | words added | slide that dish over to the front of the stove | 2 | slide that flat dish over to the front of the stove | 0 | -2 | 0.3149 |
| libero_goal/5 |  | noun/other | slide that dish over to the front of the stove | 2 | slide that plate over to the front of the stove | 62 | **+60** | <0.0001 |
| libero_goal/5 |  | noun/other | slide that flat dish over to the front of the stove | 0 | slide that plate over to the front of the stove | 62 | **+62** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the electric stove | 100 | switch on the hot plate | 2 | **-98** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the cooker | 84 | switch on the electric hot plate | 2 | **-82** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the cooker | 84 | switch on the hot plate | 2 | **-82** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the hob | 78 | switch on the hot plate | 2 | **-76** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the cooktop | 74 | switch on the electric hot plate | 2 | **-72** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the cooktop | 74 | switch on the hot plate | 2 | **-72** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the electric stove | 100 | switch on the hotplate | 30 | **-70** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the heating element | 66 | switch on the hot plate | 2 | **-64** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the burner | 64 | switch on the electric hot plate | 2 | **-62** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the burner | 64 | switch on the hot plate | 2 | **-62** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the griddle | 62 | switch on the hot plate | 2 | **-60** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the cooker | 84 | switch on the hotplate | 30 | **-54** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the hob | 78 | switch on the hotplate | 30 | **-48** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the cooktop | 74 | switch on the hotplate | 30 | **-44** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the electric stove | 100 | switch on the griddle | 62 | **-38** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the heating element | 66 | switch on the hotplate | 30 | **-36** | 0.0003 |
| libero_goal/7 |  | noun/other | switch on the burner | 64 | switch on the hotplate | 30 | **-34** | 0.0007 |
| libero_goal/7 |  | noun/other | switch on the electric stove | 100 | switch on the heating element | 66 | **-34** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the griddle | 62 | switch on the hotplate | 30 | **-32** | 0.0013 |
| libero_goal/7 |  | noun/other | switch on the cooker | 84 | switch on the griddle | 62 | **-22** | 0.0132 |
| libero_goal/7 |  | noun/other | switch on the electric stove | 100 | switch on the hob | 78 | **-22** | 0.0004 |
| libero_goal/7 |  | noun/other | switch on the cooker | 84 | switch on the heating element | 66 | **-18** | 0.0377 |
| libero_goal/7 |  | noun/other | turn on the burner | 100 | turn the black knob to activate the burner | 84 | -16 | 0.0032 |
| libero_goal/7 |  | noun/other | switch on the electric stove | 100 | switch on the range | 86 | -14 | 0.0061 |
| libero_goal/7 |  | noun/other | turn on the burner | 100 | turn the black knob to start the burner | 86 | -14 | 0.0061 |
| libero_goal/7 |  | noun/other | switch on the cooktop | 74 | switch on the griddle | 62 | -12 | 0.1984 |
| libero_goal/7 |  | noun/other | switch on the cooker | 84 | switch on the cooktop | 74 | -10 | 0.2196 |
| libero_goal/7 |  | noun/other | switch on the cooktop | 74 | switch on the heating element | 66 | -8 | 0.3827 |
| libero_goal/7 |  | noun/other | switch on the cooker | 84 | switch on the hob | 78 | -6 | 0.4444 |
| libero_goal/7 |  | noun/other | switch on the burner | 64 | switch on the griddle | 62 | -2 | 0.8359 |
| libero_goal/7 |  | noun/other | could you turn on the stove | 100 | start the stove | 100 | +0 | 1.0000 |
| libero_goal/7 |  | noun/other | could you turn on the stove | 100 | switch on the stove | 100 | +0 | 1.0000 |
| libero_goal/7 |  | words added | turn on the stove | 100 | could you turn on the stove | 100 | +0 | 1.0000 |
| libero_goal/7 |  | verb/frame | start the stove | 100 | switch on the stove | 100 | +0 | 1.0000 |
| libero_goal/7 |  | verb/frame | turn on the stove | 100 | start the stove | 100 | +0 | 1.0000 |
| libero_goal/7 |  | words removed | switch on the electric hot plate | 2 | switch on the hot plate | 2 | +0 | 1.0000 |
| libero_goal/7 |  | words removed | switch on the electric stove | 100 | switch on the stove | 100 | +0 | 1.0000 |
| libero_goal/7 |  | noun/other | switch on the electric stove | 100 | switch on the stovetop | 100 | +0 | 1.0000 |
| libero_goal/7 |  | noun/other | switch on the stove | 100 | switch on the stovetop | 100 | +0 | 1.0000 |
| libero_goal/7 |  | verb/frame | turn on the stove | 100 | switch on the stove | 100 | +0 | 1.0000 |
| libero_goal/7 |  | noun/other | turn on the burner | 100 | turn on the burners | 100 | +0 | 1.0000 |
| libero_goal/7 |  | noun/other | turn on the stove | 100 | turn on the burner | 100 | +0 | 1.0000 |
| libero_goal/7 |  | noun/other | turn on the stove | 100 | turn on the burners | 100 | +0 | 1.0000 |
| libero_goal/7 |  | verb/frame | power on the hot plate | 0 | switch on the hot plate | 2 | +2 | 0.3149 |
| libero_goal/7 |  | noun/other | switch on the burner | 64 | switch on the heating element | 66 | +2 | 0.8339 |
| libero_goal/7 |  | noun/other | switch on the cooker | 84 | switch on the range | 86 | +2 | 0.7794 |
| libero_goal/7 |  | verb/frame | turn the black knob to activate the burner | 84 | turn the black knob to start the burner | 86 | +2 | 0.7794 |
| libero_goal/7 |  | noun/other | switch on the cooktop | 74 | switch on the hob | 78 | +4 | 0.6396 |
| libero_goal/7 |  | noun/other | switch on the griddle | 62 | switch on the heating element | 66 | +4 | 0.6769 |
| libero_goal/7 |  | noun/other | start the burner | 40 | start the burners | 48 | +8 | 0.4203 |
| libero_goal/7 |  | noun/other | switch on the hob | 78 | switch on the range | 86 | +8 | 0.2978 |
| libero_goal/7 |  | noun/other | switch on the burner | 64 | switch on the cooktop | 74 | +10 | 0.2797 |
| libero_goal/7 |  | noun/other | switch on the cooktop | 74 | switch on the range | 86 | +12 | 0.1336 |
| libero_goal/7 |  | noun/other | switch on the heating element | 66 | switch on the hob | 78 | +12 | 0.1814 |
| libero_goal/7 |  | noun/other | switch on the burner | 64 | switch on the hob | 78 | +14 | 0.1229 |
| libero_goal/7 |  | noun/other | switch on the range | 86 | switch on the stove | 100 | +14 | 0.0061 |
| libero_goal/7 |  | noun/other | switch on the range | 86 | switch on the stovetop | 100 | +14 | 0.0061 |
| libero_goal/7 |  | noun/other | switch on the cooker | 84 | switch on the electric stove | 100 | +16 | 0.0032 |
| libero_goal/7 |  | noun/other | switch on the cooker | 84 | switch on the stove | 100 | +16 | 0.0032 |
| libero_goal/7 |  | noun/other | switch on the cooker | 84 | switch on the stovetop | 100 | +16 | 0.0032 |
| libero_goal/7 |  | noun/other | switch on the griddle | 62 | switch on the hob | 78 | +16 | 0.0809 |
| libero_goal/7 |  | noun/other | switch on the burner | 64 | switch on the cooker | 84 | **+20** | 0.0226 |
| libero_goal/7 |  | noun/other | switch on the burner | 64 | turn the black knob to activate the burner | 84 | **+20** | 0.0226 |
| libero_goal/7 |  | noun/other | switch on the heating element | 66 | switch on the range | 86 | **+20** | 0.0192 |
| libero_goal/7 |  | noun/other | switch on the burner | 64 | switch on the range | 86 | **+22** | 0.0111 |
| libero_goal/7 |  | noun/other | switch on the burner | 64 | turn the black knob to start the burner | 86 | **+22** | 0.0111 |
| libero_goal/7 |  | noun/other | switch on the hob | 78 | switch on the stove | 100 | **+22** | 0.0004 |
| libero_goal/7 |  | noun/other | switch on the hob | 78 | switch on the stovetop | 100 | **+22** | 0.0004 |
| libero_goal/7 |  | verb/frame | start the burner | 40 | switch on the burner | 64 | **+24** | 0.0163 |
| libero_goal/7 |  | noun/other | switch on the griddle | 62 | switch on the range | 86 | **+24** | 0.0062 |
| libero_goal/7 |  | noun/other | switch on the cooktop | 74 | switch on the electric stove | 100 | **+26** | 0.0001 |
| libero_goal/7 |  | noun/other | switch on the cooktop | 74 | switch on the stove | 100 | **+26** | 0.0001 |
| libero_goal/7 |  | noun/other | switch on the cooktop | 74 | switch on the stovetop | 100 | **+26** | 0.0001 |
| libero_goal/7 |  | noun/other | switch on the electric hot plate | 2 | switch on the hotplate | 30 | **+28** | 0.0001 |
| libero_goal/7 |  | noun/other | switch on the hot plate | 2 | switch on the hotplate | 30 | **+28** | 0.0001 |
| libero_goal/7 |  | noun/other | switch on the heating element | 66 | switch on the stove | 100 | **+34** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the heating element | 66 | switch on the stovetop | 100 | **+34** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the burner | 64 | switch on the electric stove | 100 | **+36** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the burner | 64 | switch on the stove | 100 | **+36** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the burner | 64 | switch on the stovetop | 100 | **+36** | <0.0001 |
| libero_goal/7 |  | verb/frame | switch on the burner | 64 | turn on the burner | 100 | **+36** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the griddle | 62 | switch on the stove | 100 | **+38** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the griddle | 62 | switch on the stovetop | 100 | **+38** | <0.0001 |
| libero_goal/7 |  | noun/other | start the burner | 40 | turn the black knob to activate the burner | 84 | **+44** | <0.0001 |
| libero_goal/7 |  | words added | start the burner | 40 | turn the black knob to start the burner | 86 | **+46** | <0.0001 |
| libero_goal/7 |  | noun/other | start the burners | 48 | start the stove | 100 | **+52** | <0.0001 |
| libero_goal/7 |  | verb/frame | start the burners | 48 | turn on the burners | 100 | **+52** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the hotplate | 30 | switch on the range | 86 | **+56** | <0.0001 |
| libero_goal/7 |  | noun/other | start the burner | 40 | start the stove | 100 | **+60** | <0.0001 |
| libero_goal/7 |  | verb/frame | start the burner | 40 | turn on the burner | 100 | **+60** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the electric hot plate | 2 | switch on the griddle | 62 | **+60** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the electric hot plate | 2 | switch on the heating element | 66 | **+64** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the hotplate | 30 | switch on the stove | 100 | **+70** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the hotplate | 30 | switch on the stovetop | 100 | **+70** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the electric hot plate | 2 | switch on the hob | 78 | **+76** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the electric hot plate | 2 | switch on the range | 86 | **+84** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the hot plate | 2 | switch on the range | 86 | **+84** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the electric hot plate | 2 | switch on the electric stove | 100 | **+98** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the electric hot plate | 2 | switch on the stove | 100 | **+98** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the electric hot plate | 2 | switch on the stovetop | 100 | **+98** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the hot plate | 2 | switch on the stove | 100 | **+98** | <0.0001 |
| libero_goal/7 |  | noun/other | switch on the hot plate | 2 | switch on the stovetop | 100 | **+98** | <0.0001 |
| libero_goal/7 |  | noun/other | power on the hot plate | 0 | power on the stove burner | 100 | **+100** | <0.0001 |
| libero_goal/8 |  | noun/other | put the bowl on the plate | 100 | put the bowl on the flat dish | 36 | **-64** | <0.0001 |
| libero_goal/8 |  | noun/other | put the bowl on the plate | 100 | put the bowl on the dish | 48 | **-52** | <0.0001 |
| libero_goal/8 |  | words added | put the bowl on the dish | 48 | put the bowl on the flat dish | 36 | -12 | 0.2241 |
| libero_goal/8 |  | verb/frame | Put the bowl on the plate | 100 | can you put the bowl on the plate | 100 | +0 | 1.0000 |
| libero_goal/8 |  | verb/frame | Put the bowl on the plate | 100 | i want the bowl on the plate | 100 | +0 | 1.0000 |
| libero_goal/8 |  | case | put the bowl on the plate | 100 | Put the bowl on the plate | 100 | +0 | 1.0000 |
| libero_goal/8 |  | verb/frame | can you put the bowl on the plate | 100 | i want the bowl on the plate | 100 | +0 | 1.0000 |
| libero_goal/8 |  | verb/frame | put the bowl on the plate | 100 | can you put the bowl on the plate | 100 | +0 | 1.0000 |
| libero_goal/8 |  | verb/frame | put the bowl on the plate | 100 | i want the bowl on the plate | 100 | +0 | 1.0000 |
| libero_goal/8 |  | punctuation | put the bowl on the plate | 100 | put the bowl on the plate. | 100 | +0 | 1.0000 |
| libero_goal/8 |  | noun/other | put the bowl on the plate | 100 | put the dish on the plate | 100 | +0 | 1.0000 |
| libero_goal/8 |  | verb/frame | the bowl goes on the plate | 100 | the bowl should be put on the plate | 100 | +0 | 1.0000 |
| libero_goal/8 |  | noun/other | put the bowl on the dish | 48 | put the bowl on the plate. | 100 | **+52** | <0.0001 |
| libero_goal/8 |  | noun/other | put the bowl on the flat dish | 36 | put the bowl on the plate. | 100 | **+64** | <0.0001 |
| libero_goal/9 |  | noun/other | put the wine bottle on the rack | 94 | put the wine bottle on the wooden stand | 50 | **-44** | <0.0001 |
| libero_goal/9 |  | color | put the wine bottle on the stand | 76 | put the wine bottle on the wooden stand | 50 | **-26** | 0.0071 |
| libero_goal/9 |  | noun/other | put the wine bottle on the rack | 94 | put the wine bottle on the stand | 76 | **-18** | 0.0117 |
| libero_goal/9 |  | noun/other | put the wine bottle on the rack | 94 | put the wine jug on the rack | 98 | +4 | 0.3074 |
| libero_object/0 |  | noun/other | pick up the alphabet soup and place it in the basket | 100 | pick up the food and place it in the basket | 86 | -14 | 0.0061 |
| libero_object/0 |  | noun/other | pick up the alphabet soup and place it in the basket | 100 | pick up the tin and place it in the basket | 90 | -10 | 0.0218 |
| libero_object/0 |  | noun/other | pick up the soup and place it in the basket | 100 | pick up the tin and place it in the basket | 90 | -10 | 0.0218 |
| libero_object/0 |  | noun/other | pick up the alphabet soup and place it in the basket | 100 | pick up the can and place it in the basket | 92 | -8 | 0.0412 |
| libero_object/0 |  | noun/other | pick up the can and place it in the basket | 92 | pick up the food and place it in the basket | 86 | -6 | 0.3377 |
| libero_object/0 |  | noun/other | pick up the can and place it in the basket | 92 | pick up the tin and place it in the basket | 90 | -2 | 0.7268 |
| libero_object/0 |  | words removed | pick up the alphabet soup and place it in the basket | 100 | pick up the soup and place it in the basket | 100 | +0 | 1.0000 |
| libero_object/0 |  | noun/other | pick up the food and place it in the basket | 86 | pick up the tin and place it in the basket | 90 | +4 | 0.5383 |
| libero_object/0 |  | noun/other | pick up the can and place it in the basket | 92 | pick up the soup and place it in the basket | 100 | +8 | 0.0412 |
| libero_object/0 |  | noun/other | pick up the food and place it in the basket | 86 | pick up the soup and place it in the basket | 100 | +14 | 0.0061 |
| libero_object/1 | yes | noun/other | pick up the cheese and place it in the basket | 100 | pick up the container and place it in the basket | 34 | **-66** | <0.0001 |
| libero_object/1 | yes | noun/other | pick up the cream cheese and place it in the basket | 100 | pick up the container and place it in the basket | 34 | **-66** | <0.0001 |
| libero_object/1 | yes | noun/other | pick up the box and place it in the basket | 96 | pick up the container and place it in the basket | 34 | **-62** | <0.0001 |
| libero_object/1 | yes | noun/other | pick up the cheese and place it in the basket | 100 | pick up the package and place it in the basket | 84 | -16 | 0.0032 |
| libero_object/1 | yes | noun/other | pick up the cream cheese and place it in the basket | 100 | pick up the package and place it in the basket | 84 | -16 | 0.0032 |
| libero_object/1 | yes | noun/other | pick up the box and place it in the basket | 96 | pick up the package and place it in the basket | 84 | -12 | 0.0455 |
| libero_object/1 | yes | noun/other | pick up the cream cheese and place it in the basket | 100 | pick up the box and place it in the basket | 96 | -4 | 0.1531 |
| libero_object/1 | yes | verb/frame | Pick up the cream cheese and place it in the basket | 100 | grab the cream cheese and place it in the basket | 100 | +0 | 1.0000 |
| libero_object/1 | yes | verb/frame | Pick up the cream cheese and place it in the basket | 100 | lift the cream cheese and place it in the basket | 100 | +0 | 1.0000 |
| libero_object/1 | yes | case | pick up the cream cheese and place it in the basket | 100 | Pick up the cream cheese and place it in the basket | 100 | +0 | 1.0000 |
| libero_object/1 | yes | verb/frame | grab the cream cheese and place it in the basket | 100 | lift the cream cheese and place it in the basket | 100 | +0 | 1.0000 |
| libero_object/1 | yes | verb/frame | pick up the cream cheese and place it in the basket | 100 | grab the cream cheese and place it in the basket | 100 | +0 | 1.0000 |
| libero_object/1 | yes | verb/frame | pick up the cream cheese and place it in the basket | 100 | lift the cream cheese and place it in the basket | 100 | +0 | 1.0000 |
| libero_object/1 | yes | words removed | pick up the cream cheese and place it in the basket | 100 | pick up the cheese and place it in the basket | 100 | +0 | 1.0000 |
| libero_object/1 | yes | punctuation | pick up the cream cheese and place it in the basket | 100 | pick up the cream cheese and place it in the basket. | 100 | +0 | 1.0000 |
| libero_object/1 | yes | verb/frame | pick up the cream cheese and place it in the basket | 100 | pick up the cream cheese and put it in the basket | 100 | +0 | 1.0000 |
| libero_object/1 | yes | noun/other | pick up the box and place it in the basket | 96 | pick up the cheese and place it in the basket | 100 | +4 | 0.1531 |
| libero_object/1 | yes | noun/other | pick up the container and place it in the basket | 34 | pick up the package and place it in the basket | 84 | **+50** | <0.0001 |
| libero_object/2 |  | noun/other | pick up the salad dressing and place it in the basket | 100 | pick up the bottle and place it in the basket | 74 | **-26** | 0.0001 |
| libero_object/2 |  | verb/frame | pick up the salad dressing and place it in the basket | 100 | lift the salad dressing and place it in the basket | 98 | -2 | 0.3149 |
| libero_object/2 |  | words removed | pick up the salad dressing and place it in the basket | 100 | pick up the dressing and place it in the basket | 100 | +0 | 1.0000 |
| libero_object/2 |  | noun/other | pick up the bottle and place it in the basket | 74 | pick up the dressing and place it in the basket | 100 | **+26** | 0.0001 |
| libero_object/3 |  | noun/other | pick up the bbq sauce and place it in the basket | 98 | pick up the bottle and place it in the basket | 60 | **-38** | <0.0001 |
| libero_object/3 |  | noun/other | pick up the bbq sauce and place it in the basket | 98 | pick up the condiment and place it in the basket | 82 | -16 | 0.0077 |
| libero_object/3 |  | noun/other | pick up the bottle and place it in the basket | 60 | pick up the condiment and place it in the basket | 82 | **+22** | 0.0153 |
| libero_object/4 |  | noun/other | pick up the ketchup and place it in the basket | 98 | pick up the sauce and place it in the basket | 2 | **-96** | <0.0001 |
| libero_object/4 |  | noun/other | pick up the bottle and place it in the basket | 96 | pick up the sauce and place it in the basket | 2 | **-94** | <0.0001 |
| libero_object/4 |  | noun/other | pick up the red bottle and place it in the basket | 58 | pick up the sauce and place it in the basket | 2 | **-56** | <0.0001 |
| libero_object/4 |  | noun/other | pick up the ketchup and place it in the basket | 98 | pick up the red bottle and place it in the basket | 58 | **-40** | <0.0001 |
| libero_object/4 |  | color | pick up the bottle and place it in the basket | 96 | pick up the red bottle and place it in the basket | 58 | **-38** | <0.0001 |
| libero_object/4 |  | noun/other | pick up the ketchup and place it in the basket | 98 | pick up the bottle and place it in the basket | 96 | -2 | 0.5577 |
| libero_object/5 |  | noun/other | pick up the tomato sauce and place it in the basket | 94 | pick up the bottle and place it in the basket | 80 | -14 | 0.0374 |
| libero_object/5 |  | noun/other | pick up the tomato sauce and place it in the basket | 94 | pick up the can and place it in the basket | 90 | -4 | 0.4610 |
| libero_object/5 |  | words removed | pick up the tomato sauce and place it in the basket | 94 | pick up the sauce and place it in the basket | 96 | +2 | 0.6464 |
| libero_object/5 |  | noun/other | pick up the can and place it in the basket | 90 | pick up the sauce and place it in the basket | 96 | +6 | 0.2397 |
| libero_object/5 |  | noun/other | pick up the bottle and place it in the basket | 80 | pick up the can and place it in the basket | 90 | +10 | 0.1614 |
| libero_object/5 |  | noun/other | pick up the bottle and place it in the basket | 80 | pick up the sauce and place it in the basket | 96 | +16 | 0.0138 |
| libero_object/6 | yes | noun/other | pick up the butter and place it in the basket | 100 | pick up the block and place it in the basket | 82 | **-18** | 0.0017 |
| libero_object/6 | yes | noun/other | pick up the butter and place it in the basket | 100 | pick up the stick and place it in the basket | 84 | -16 | 0.0032 |
| libero_object/6 | yes | noun/other | pick up the block and place it in the basket | 82 | pick up the stick and place it in the basket | 84 | +2 | 0.7901 |
| libero_object/7 |  | noun/other | pick up the milk and place it in the basket | 100 | pick up the drink and place it in the basket | 94 | -6 | 0.0786 |
| libero_object/7 |  | noun/other | pick up the milk and place it in the basket | 100 | pick up the carton and place it in the basket | 96 | -4 | 0.1531 |
| libero_object/7 |  | noun/other | pick up the carton and place it in the basket | 96 | pick up the drink and place it in the basket | 94 | -2 | 0.6464 |
| libero_object/8 |  | noun/other | pick up the chocolate pudding and place it in the basket | 100 | pick up the cup and place it in the basket | 64 | **-36** | <0.0001 |
| libero_object/8 |  | noun/other | pick up the chocolate pudding and place it in the basket | 100 | pick up the snack and place it in the basket | 96 | -4 | 0.1531 |
| libero_object/8 |  | noun/other | pick up the cup and place it in the basket | 64 | pick up the snack and place it in the basket | 96 | **+32** | <0.0001 |
| libero_object/9 |  | noun/other | pick up the orange juice and place it in the basket | 98 | pick up the bottle and place it in the basket | 12 | **-86** | <0.0001 |
| libero_object/9 |  | noun/other | pick up the orange juice and place it in the basket | 98 | pick up the carton and place it in the basket | 12 | **-86** | <0.0001 |
| libero_object/9 |  | noun/other | pick up the orange juice and place it in the basket | 98 | pick up the drink and place it in the basket | 20 | **-78** | <0.0001 |
| libero_object/9 |  | color | pick up the orange juice and place it in the basket | 98 | pick up the juice and place it in the basket | 96 | -2 | 0.5577 |
| libero_object/9 |  | noun/other | pick up the bottle and place it in the basket | 12 | pick up the carton and place it in the basket | 12 | +0 | 1.0000 |
| libero_object/9 |  | noun/other | pick up the bottle and place it in the basket | 12 | pick up the drink and place it in the basket | 20 | +8 | 0.2752 |
| libero_object/9 |  | noun/other | pick up the carton and place it in the basket | 12 | pick up the drink and place it in the basket | 20 | +8 | 0.2752 |
| libero_object/9 |  | noun/other | pick up the drink and place it in the basket | 20 | pick up the juice and place it in the basket | 96 | **+76** | <0.0001 |
| libero_object/9 |  | noun/other | pick up the bottle and place it in the basket | 12 | pick up the juice and place it in the basket | 96 | **+84** | <0.0001 |
| libero_object/9 |  | noun/other | pick up the carton and place it in the basket | 12 | pick up the juice and place it in the basket | 96 | **+84** | <0.0001 |
| libero_spatial/1 |  | noun/other | pick up the black bowl next to the ramekin and place it on the plate | 100 | pick up the black bowl next to the dish and place it on the plate | 62 | **-38** | <0.0001 |
| libero_spatial/3 | yes | noun/other | pick up the black bowl on the cookie box and place it on the plate | 100 | pick up the black bowl on the cookie tin and place it on the plate | 96 | -4 | 0.1531 |
| libero_spatial/3 | yes | noun/other | pick up the black bowl on the cookie box and place it on the plate | 100 | pick up the black bowl on the cookies and place it on the plate | 100 | +0 | 1.0000 |
| libero_spatial/3 | yes | noun/other | pick up the black bowl on the cookie tin and place it on the plate | 96 | pick up the black bowl on the cookies and place it on the plate | 100 | +4 | 0.1531 |
| libero_spatial/4 |  | noun/other | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 90 | pick up the black bowl in the upper drawer of the wooden cabinet and place it on the plate | 90 | +0 | 1.0000 |
| libero_spatial/4 |  | color | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 90 | pick up the black bowl in the top drawer of the cabinet and place it on the plate | 96 | +6 | 0.2397 |
| libero_spatial/5 |  | noun/other | pick up the black bowl on the ramekin and place it on the plate | 96 | pick up the black bowl on the dish and place it on the plate | 86 | -10 | 0.0806 |
| libero_spatial/5 |  | preposition/particle | pick up the black bowl on the ramekin and place it on the plate | 96 | pick up the black bowl on the ramekin and place it onto the plate | 92 | -4 | 0.3997 |
| libero_spatial/5 |  | noun/other | pick up the black bowl on the ramekin and place it on the plate | 96 | pick up the black bowl on the ramekin and place it on the dish | 96 | +0 | 1.0000 |
| libero_spatial/5 |  | noun/other | pick up the black bowl on the ramekin and place it on the plate | 96 | pick up the dark bowl on the ramekin and place it on the plate | 98 | +2 | 0.5577 |
| libero_spatial/5 |  | verb/frame | pick up the black bowl on the ramekin and place it on the plate | 96 | lift the black bowl on the ramekin and place it on the plate | 100 | +4 | 0.1531 |
| libero_spatial/6 | yes | noun/other | pick up the black bowl next to the cookie box and place it on the plate | 100 | pick up the black bowl next to the cookies and place it on the plate | 98 | -2 | 0.3149 |
| libero_spatial/6 | yes | preposition/particle | pick up the black bowl next to the cookie box and place it on the plate | 100 | pick up the black bowl beside the cookie box and place it on the plate | 100 | +0 | 1.0000 |
| libero_spatial/7 | yes | noun/other | pick up the black bowl on the stove and place it on the plate | 100 | pick up the black bowl on the hot plate and place it on the plate | 22 | **-78** | <0.0001 |
| libero_spatial/8 |  | preposition/particle | pick up the black bowl next to the plate and place it on the plate | 98 | pick up the black bowl beside the plate and place it on the plate | 100 | +2 | 0.3149 |
| libero_spatial/9 |  | color | pick up the black bowl on the wooden cabinet and place it on the plate | 98 | pick up the black bowl on the cabinet and place it on the plate | 94 | -4 | 0.3074 |
| libero_spatial/9 |  | noun/other | pick up the black bowl on the wooden cabinet and place it on the plate | 98 | pick up the black bowl on the wood cabinet and place it on the plate | 96 | -2 | 0.5577 |
| libero_spatial/9 |  | words added | pick up the black bowl on the cabinet and place it on the plate | 94 | pick up the black bowl on the wood cabinet and place it on the plate | 96 | +2 | 0.6464 |
