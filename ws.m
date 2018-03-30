clc, clear;
close all;

rgb = imread('../Tissue images/TCGA-21-5784-01Z-00-DX1.tif');
I = rgb2gray(rgb);

hy = fspecial('sobel');
hx = hy';
Iy = imfilter(double(I), hy, 'replicate');
Ix = imfilter(double(I), hx, 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);
figure
imshow(Ix,[]), title('Ix');
figure
imshow(Iy,[]), title('Iy');
figure
imshow(gradmag,[]), title('Gradient magnitude (gradmag)');

L = watershed(gradmag);
Lrgb = label2rgb(L);
figure, imshow(Lrgb), title('Watershed transform of gradient magnitude (Lrgb)');

se = strel('disk', 20);
Io = imopen(I, se);
figure
imshow(Io), title('Opening (Io)');

Ie = imerode(I, se);
Iobr = imreconstruct(Ie, I);
figure
imshow(Iobr), title('Opening-by-reconstruction (Iobr)');

Ioc = imclose(Io, se);
figure
imshow(Ioc), title('Opening-closing (Ioc)');

Iobrd = imdilate(Iobr, se);
Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr));
Iobrcbr = imcomplement(Iobrcbr);
figure
imshow(Iobrcbr), title('Opening-closing by reconstruction (Iobrcbr)');

fgm = imregionalmax(Iobrcbr);
figure
imshow(fgm), title('Regional maxima of opening-closing by reconstruction (fgm)');

