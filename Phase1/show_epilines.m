images = imageDatastore('./');

I1 = readimage(images, 4);
I2 = readimage(images, 5);
% figure
% imshowpair(I1, I2, 'montage'); 
% title('Original Images');


% figure 
% imshowpair(I1, I2, 'montage');
% title('Undistorted Images');

% Create the point tracker
% Detect feature points
imagePoints1 = detectMinEigenFeatures(im2gray(I1), MinQuality = 0.1);

% Visualize detected points
% figure
% imshow(I1, InitialMagnification = 50);
% title('150 Strongest Corners from the First Image');
% hold on
% plot(selectStrongest(imagePoints1, 150));
% Create the point tracker
tracker = vision.PointTracker(MaxBidirectionalError=1, NumPyramidLevels=5);

% Initialize the point tracker
imagePoints1 = imagePoints1.Location;
initialize(tracker, imagePoints1, I1);

% Track the points
[imagePoints2, validIdx] = step(tracker, I2);
matchedPoints1 = imagePoints1(validIdx, :);
matchedPoints2 = imagePoints2(validIdx, :);

% Visualize correspondences
% figure
% showMatchedFeatures(I1, I2, matchedPoints1, matchedPoints2);
% title('Tracked Features');
K = [
    531.122155322710 0 407.192550839899;
0 531.541737503901 313.308715048366;
0 0 1;
    ];
KK = cameraIntrinsicsFromOpenCV(K, [0.015,0.05,0,0,0],[600,800]);
% Estimate the fundamental matrix
[E, epipolarInliers] = estimateEssentialMatrix(matchedPoints1, matchedPoints2, KK, Confidence = 99.99);
[fLMedS,inliers] = estimateFundamentalMatrix(matchedPoints1,...
    matchedPoints2,'NumTrials',4000);
% Find epipolar inliers
inlierPoints1 = matchedPoints1(epipolarInliers, :);
inlierPoints2 = matchedPoints2(epipolarInliers, :);

% Display inlier matches
% figure
% showMatchedFeatures(I1, I2, inlierPoints1, inlierPoints2);
% title('Epipolar Inliers');
figure; 
subplot(121);
imshow(I1); 
title('Inliers and Epipolar Lines in First Image'); hold on;
plot(matchedPoints1(epipolarInliers,1),matchedPoints1(epipolarInliers,2),'go')


epiLines = epipolarLine(fLMedS',matchedPoints2(inliers,:));
points = lineToBorderPoints(epiLines,size(I1));
line(points(:,[1,3])',points(:,[2,4])');

subplot(122); 
imshow(I2);
title('Inliers and Epipolar Lines in Second Image'); hold on;
plot(matchedPoints2(inliers,1),matchedPoints2(inliers,2),'go')
epiLines = epipolarLine(fLMedS,matchedPoints1(inliers,:));
points = lineToBorderPoints(epiLines,size(I2));
line(points(:,[1,3])',points(:,[2,4])');
truesize;