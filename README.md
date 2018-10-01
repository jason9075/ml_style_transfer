# Transfer Learning 藝術風格轉移

此專案是為了瞭解機器學習如何擷取特徵風格的練習，原型使用 Keras VGG 16為原型，但實測下來結果發現只取用前12層效果較佳，過少層風格會不明顯，過多層則會導致圖片原型扭曲變形。  

底下分別為套用結果：

### 原圖：
 <img src="https://raw.githubusercontent.com/jason9075/ml_style_transfer/master/demo_result/me.jpg" height="360">


風格             |  結果
:-------------------------:|:-------------------------:
<img src="https://raw.githubusercontent.com/jason9075/ml_style_transfer/master/styles/monalisa.jpg" height="360">    | <img src="https://raw.githubusercontent.com/jason9075/ml_style_transfer/master/demo_result/monalisa.jpg" height="360">
<img src="https://raw.githubusercontent.com/jason9075/ml_style_transfer/master/styles/scream.jpg" height="360">      | <img src="https://raw.githubusercontent.com/jason9075/ml_style_transfer/master/demo_result/scream.jpg" height="360">
<img src="https://raw.githubusercontent.com/jason9075/ml_style_transfer/master/styles/starrynight.jpg" height="360"> | <img src="https://raw.githubusercontent.com/jason9075/ml_style_transfer/master/demo_result/starrynight.jpg" height="360">
