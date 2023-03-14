## Build Documentation

1. Clone mindface

   ```bash
   git clone https://openi.pcl.ac.cn/OpenModelZoo/MindFace_Detection.git
   cd mindface
   ```

2. Install the building dependencies of documentation

   ```bash
   pip install -r docs/detection/requirements.txt
   ```

3. Change directory to `docs` 

   ```bash
   cd docs
   ```

4. Build documentation

   ```bash
   sphinx-build -b html en build  
   ```

5. Open `build/index.html` with browser