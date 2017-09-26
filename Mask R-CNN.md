## Mask R-CNN

            Faster R-CNN 와 마찬가지로 영상에서 여러 ROI 후보를 제안한다. 제안된 ROI 후보는 곧 경계 박스이며 해당 위치의 특징맵을 RoIAlign 방식으로 추출한다. 추출된 특징맵으로부터 오브젝트의 클래스를 분류함과 동시에 오브젝트의 마스크를 얻는다

 원본 영상의 좌상단의 15x15 영역이 RoI (= 경계박스)다. CNN은 원본 이미지 128x128 을 입력 받아 25 x 25 의 특징맵을 출력한다. 원본 영상의 15 x 15 에 해당하는 특징맵의 크기는 2.93 x 2.93 이다. 128 x 128 이 25 x 25 로 작아졌다. 128 / 5.12 = 25 15 / 5.12 = 2.93 이므로 원본 이미지의 15 x 15 는 최종 특징 맵의 2.93 x 2.93 이다.

 RoIPool 은 이런 경우에 2.93을 반올림 하여 특징맵 좌상단의 3x3 부분만 가져와서(pool) 클래스를 예측하였다. 2.93은 반올림되어 3.00 이 된다. 0.07의 차이는 작아보일지 모르지만 이 차이는 최대 0.5 까지 발생하며, 이러한 정렬 불량 은 성능에 큰 영향을 미친다. RoIAlign 은 2.93 x 2.93 에 해당하는 특징맵을 정렬(align)시켜 바이리니어 인터폴레이션을 사용하여 보정된 특징값을 사용한다.

 RoIPool 과 RoIWarp는 align 에 대한 고려가 없이 반올림을 이용해서 특징맵의 RoI 영역을 가져온다. RoIWarp 는 반올림을 이용해 특징맵의 RoI를 가져온 후에 bilinear 인터폴레이션을 이용해서 지정된 크기로 resize 한다. 그렇기 때문에 정렬 상태 불량(misalignment) 현상이 발생한다. RoiAlign 은 반올림 등을 사용하지 않고 bilinear 인터폴레이션을 이용해서 특징맵의 RoI 영역을 정확하게 정렬되도록 한다.

 그리고 올해 초 페이스북 AI 팀이 분할된 이미지를 마스킹하는 Mask R-CNN을 내놓았습니다. Faster R-CNN에 각 픽셀이 오브젝트에 해당하는 것인지 아닌지를 마스킹하는 네트워크(CNN)를 추가한 것입니다. 이를 바이너리 마스크binary mask라고 합니다. 페이스북 팀은 정확한 픽셀 위치를 추출하기 위해 CNN을 통과하면서 RoIPool 영역의 위치에 생기는 소숫점 오차를 2D 선형보간법bilinear interpolation을 통해 감소시켰다고 합니다. 이를 RoIAlign이라고 합니다.

> Interpolation(인터폴레이션, 보간)이란 알려진 지점의 값 사이(중간)에 위치한 값을 알려진 값으로부터 추정하는 것을 말한다.

RoIAlign
* Avoid any quantization, realizing alignment between mask and instance(e.g. use x/16x/16 instead of [x/16][x/16])
* Use bilinear interpolation on feature map
* Compared to RoIWarp, which also adopts bilinear resampling, proving the effectiveness of RoIAlign mainly comes from alignment instead of bilinear interpolation.
