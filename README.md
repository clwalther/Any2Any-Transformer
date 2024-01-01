# `Any2Any-Transformer`
`Any2Any-Transformer` is a transformer project aimed at exploring the possibility of using
different encoders and decoders while not loosing context.
<br>
This is of interest as one might be able to cut down on hard to aquire
(and therefore expensive) datasets for such a models.


## Table of Contents
- [Table of Contents](https://github.com/clwalther/Any2Any-Transformer#Table-of-Contents)
- [Disclaimer](https://github.com/clwalther/Any2Any-Transformer#Disclaimer)
- [Abstract](https://github.com/clwalther/Any2Any-Transformer#Abstract)
- [Ressources](https://github.com/clwalther/Any2Any-Transformer#Ressources)


## Disclaimer
Due to the limited support by [tensorflow-text](https://github.com/tensorflow/text)[^1]
the code in this repository only runs
on ["Linux x86_64 and Intel-based Macs"](https://github.com/tensorflow/text#a-note-about-different-operating-system-packages)[^2].
<br>
You can try to build the required version of tensorflow-text yourself.
<br>
If this isn't an opption for you I found that Github Codespaces runs on the required
Linux x86 64 bit and therefore is suitable for this application.
<br>
I, myself have to resort to developing in this enviornment.


## Abstract
The code you will see here is inspired by this
[tutorial](https://www.tensorflow.org/text/tutorials/transformer)[^3] by the
Tensorflow-Team, covering the in 2018 proposed transformer-model.
<br>
More specifically [Mark Daoust](https://github.com/MarkDaoust) and ofcourse the authors
of the underlaying paper ["Attention is all you need"](https://arxiv.org/pdf/1706.03762.pdf)[^4]
are people I want to give credit to.

---

Here I want to explore the possibility of using numerous encoders and decoders to
potentially achive better training performance and a way of understanding and producing
any kind of media.
<br>
(In this context a medium is understood as to be the means by which something is comunicated
or expressed, including but not limited to Text (e.g.: in different languages), Images, Sound
(e.g.: Speech, Music), Videos and the like.)


## Ressources
To achive better readabilty any ressources are linked in the text and again here in the following:

[^1]: https://github.com/tensorflow/text (last accessed: 31st of December 2023)
[^2]: https://github.com/tensorflow/text#a-note-about-different-operating-system-packages (last accessed: 31st of December 2023)
[^3]: https://www.tensorflow.org/text/tutorials/transformer (last accessed: 31st of December 2023)
[^4]: https://arxiv.org/pdf/1706.03762.pdf (last accessed: 31st of December 2023)
