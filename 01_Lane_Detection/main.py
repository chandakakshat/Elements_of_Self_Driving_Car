import tkinter 
import matplotlib.pyplot as plt
import cv2
import os
from os.path import join, basename
from collections import deque
from lane_detection import lane_detection_pipeline


if __name__ == '__main__':

    resize_height, resize_width = 540, 960

    verbose = True
    if verbose:
        plt.ion() #turn on interactive mode
        figure_manager = plt.get_current_fig_manager()
        figure_manager.window.showMaximized()

    # test on images
    test_images_dir = join('data', 'test_images')
    test_images = [join(test_images_dir, name) for name in os.listdir(test_images_dir)]

    for test_img in test_images:

        print('Processing image: {}'.format(test_img))

        out_path = join('out', 'images', basename(test_img))
        input_image = cv2.cvtColor(cv2.imread(test_img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        output_image = lane_detection_pipeline([input_image], solid_lines=True)
        cv2.imwrite(out_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        if verbose:
            plt.imshow(output_image)
            plt.waitforbuttonpress()
    plt.close('all')

    #test on videos
    test_videos_dir = join('data', 'test_videos')
    test_videos = [join(test_videos_dir, name) for name in os.listdir(test_videos_dir)]

    for test_video in test_videos:

        print('Processing video: {}'.format(test_video))

        cap = cv2.VideoCapture(test_video)
        out = cv2.VideoWriter(join('out', 'videos', basename(test_video)),
                              fourcc=cv2.VideoWriter_fourcc(*'DIVX'),
                              fps=20.0, frameSize=(resize_width, resize_height))

        frame_buffer = deque(maxlen=10)
        while cap.isOpened():
            ret, color_frame = cap.read()
            if ret:
                color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                color_frame = cv2.resize(color_frame, (resize_width, resize_height))
                frame_buffer.append(color_frame)
                blend_frame = lane_detection_pipeline(frames=frame_buffer, solid_lines=True, temporal_smoothing=True)
                out.write(cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR))
                cv2.imshow('blend', cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()



