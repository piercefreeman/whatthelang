from whatthelang import WhatTheLang
wtl = WhatTheLang()

print(wtl.predict_lang("Mother"))
print(wtl.pred_prob("Mother"))

print(wtl.predict_lang(["Mother", "Other"]))
print(wtl.pred_prob(["Mother pero maybe otra lingua", "Other"]))

assert wtl.predict_lang("Mother") == 'en'

assert wtl.predict_lang("தாய்") == 'ta'

assert wtl.predict_lang("അമ്മ") == 'ml'

assert wtl.predict_lang("पिता") == 'hi'
