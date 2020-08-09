import os
#Using ffmpeg to convert a set of images into a video (http://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/)

#To take a list of images that are padded with zeros (pic0001.png, pic0002.png…. etc) use the following command:
#ffmpeg -r 30 -f image2 -s 1920x1080 -i pic%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
# If no padding is needed use something similar to pic%d.png or %d.png.
#libx264 codec, from experience it has given me excellent quality for small video sizes.
# -r is the framerate (fps)
# -crf is the quality, lower means better quality, 15-25 is usually good
# -s is the resolution
# -pix_fmt yuv420p specifies the pixel format, change this as needed
# 

# Specifying start and end frames
# ffmpeg -r 60 -f image2 -s 1920x1080 -start_number 1 -i pic%04d.png -vframes 1000 -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
# -start_number specifies what image to start at
# -vframes 1000 specifies the number frames/images in the video

os.system("ffmpeg -f image2 -r 30 -i ./%04d.png -vcodec libx264 -y ./out.mp4")#0001.png, 0002.png, ... 
os.system("ffmpeg -pattern_type glob -i '*.png' -vcodec libx264 -y out.mp4")

# Converting a video to mp4 from a different format
# If the video has already been compressed the following can be used to change the codmpression to h264:
ffmpeg -i 6754_dehazed_wang_125f_per_s.avi -vcodec libx264 6754_dehazed_wang_125f_per_s.mp4
