from whatthelang import WhatTheLang
wtl = WhatTheLang()

assert wtl.predict_lang("Mother") == 'en'

assert wtl.predict_lang("தாய்") == 'ta'

assert wtl.predict_lang("അമ്മ") == 'ml'

assert wtl.predict_lang("पिता") == 'hi'
