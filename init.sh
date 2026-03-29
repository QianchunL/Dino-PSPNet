 #!/bin/bash
 python -c "                                                                                                                                 
  import torchvision.datasets as D                                                                                                            
  D.VOCSegmentation('./data', year='2012', image_set='train', download=True)
  " 