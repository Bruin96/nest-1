import argparse
import os
import sys
import subprocess
import socket
import json
import math
import pickle
import random
import time
from textwrap import dedent
from typing import NoReturn, Sequence, Union

import numpy as np
import psutil
from psychopy import core, data, event, gui, monitors, visual  # , clock
from psychopy.tools.filetools import fromFile, toFile
from psychopy.tools.monitorunittools import cm2pix, deg2pix, pix2deg
from psychopy.visual import Circle, Rect

from lib.additional_staircase import ExtendedMultiStairHandler
from lib.dct_tools import generate_stimulus, visualize_stimuli
from lib.display_tools import supported_displays, get_display_params
from lib.utils import reset_display

p = psutil.Process(os.getpid())
if os.name == "nt":
    p.nice(psutil.HIGH_PRIORITY_CLASS)
else:
    print(
        f"WARNING: OS {os.name} requires elevated permissions to raise the process priority!"
    )


def compute_windowed_Fisher_energy(energies, window=10):
    N = energies.shape[0]
    Fisher_diff = np.zeros(N-1)
    for i in range(1, num_samples):
        Fisher_diff[i-1] = np.abs(Fisher_energies[:, i]	- Fisher_energies[:, i-1])	
        
    Fisher_window_diff = np.array([np.sum(np.array([1/window*Fisher_diff[k-j] \
		for j in range(window)])) for k in range(window-1, len(Fisher_diff))])
    
    windowed_value = Fisher_window_diff[-1]
    return windowed_value

# from psychopy.monitors import MonitorCenter
# import matplotlib.pyplot as plt


def fixation_target(
        diameter: float,
        win: visual.Window,
        fg: Sequence[float] = (-1, -1, -1),
        bg: Sequence[float] = (0, 0, 0),
        auto_draw: bool = False,
) -> Sequence[Union[Circle, Rect]]:
    """
    Returns the fixation target whose size is specified by the given diameter.

       .-| |-.
      /__| |__\
          O
      \--| |--/
       `-| |-'
      <------->
       diameter

    Parameters:
        diameter (float): The diameter of the target.
        win (visual.Window): The window object where the target will be drawn.
        fg (Sequence[float]): Foreground color (default: (-1,-1,-1)).
        bg (Sequence[float]): Background color (default: (0,0,0)).
        auto_draw (bool): When True, the target will be drawn at each frame
            without calling draw() explicitly. (default: False)

    Returns:
        Sequence[Union[Circle, Rect]]: A tuple containing the Circle and
            Rect objects.
    """
    units = "pix"
    outer = Circle(
        win,
        radius=diameter / 2,
        edges=64,
        units=units,
        fillColor=fg,
        lineColor=bg,
    )
    band_horz = Rect(
        win,
        size=(diameter, diameter / 6),
        units=units,
        fillColor=bg,
        lineColor=bg,
    )
    band_vert = Rect(
        win,
        size=(diameter / 6, diameter),
        units=units,
        fillColor=bg,
        lineColor=bg,
    )
    inner = Circle(
        win,
        radius=diameter / 6,
        edges=64,
        units=units,
        fillColor=fg,
        lineColor=bg,
    )
    outer.autoDraw = auto_draw
    band_horz.autoDraw = auto_draw
    band_vert.autoDraw = auto_draw
    inner.autoDraw = auto_draw
    return outer, band_horz, band_vert, inner


def get_conditions(
        session_number: int,
        patch_px: Sequence[int],
        display_peak_cpd: float,
        display_fps: int,
) -> Sequence[dict]:
    total_sessions = 6
    eccentricities = (10, 25, 40)
    x = (0, (patch_px[1] - 1) // 4, (patch_px[1] - 1) // 2)
    y = (0, (patch_px[2] - 1) // 4, (patch_px[2] - 1) // 2)
    print(f"x: {x}")
    print(f"y: {y}")
    if 1 <= session_number <= 6:
        # The first set of subjective experiments
        # Tests with 20, 30, 60Hz temporal freq.
        z = (8, 12, 24)  # 20, 30, 60 Hz
    elif 7 <= session_number <= 12:
        # The second set of subjective experiments
        # Tests with 2.5, 5 and 10Hz temporal freq.
        session_number -= 6
        z = (1, 2, 4)  # 2.5, 5, 10Hz
    else:
        msg = f"Session number {session_number} is not recognized."
        raise ValueError(msg)

    all_conditions = []
    for x_ in x:
        for y_ in y:
            for z_ in z:
                for ecc in eccentricities:
                    all_conditions.append((
                        ecc,
                        x_,
                        y_,
                        z_))
    num_conditions = len(all_conditions)
    cond_per_session = num_conditions // total_sessions
    start_idx = (session_number - 1) * cond_per_session
    end_idx = session_number * cond_per_session
    if session_number == total_sessions:
        end_idx = num_conditions
    current_session = all_conditions[start_idx:end_idx]
    conditions = []
    for i in range(len(current_session)):
        conditions.append(
            {
                "label":            "ECC%d_X%d_Y%d_Z%d" % current_session[i],
                "startVal":         1 / 256 * 64,
                "stepType":         "lin",
                "method":           "2AFC",
                "stepSizes":        [
                    10 / 256,
                    7 / 256,
                    5 / 256,
                    3 / 256,
                    2 / 256,
                    ],  # steps smaller than 1/256 does not have an effect on 8 bpc
                "minVal":           1 / 256,
                "maxVal":           0.5,
                "nTrials":          120,
                "eccentricity":     current_session[i][0],
                "x":                current_session[i][1],
                "y":                current_session[i][2],
                "z":                current_session[i][3],
                "patch_size":       patch_px,
                "display_peak_cpd": display_peak_cpd,
                "display_fps":      display_fps,
            }
        )
        # TODO: add peak cpd of the display and optionally
        # the peak FPS for future reference
    return conditions


def get_flat_position(deg: float, mon: monitors.Monitor) -> float:
    dist_px = cm2pix(mon.getDistance(), mon)
    return np.tan(np.radians(deg)) * dist_px


def test_blind_spot(
        win: visual.Window,
        fixation_target: Sequence[Union[Circle, Rect]],
        diameter_px: int,
        ecc: float,
) -> NoReturn:
    patch = np.ones((diameter_px - 1, diameter_px - 1)) * -1
    mon = win.monitor
    for side in ("left", "right"):
        if side == "left":
            flip = 1
        else:
            flip = -1
        for e in ecc:
            stim = visual.ImageStim(
                win,
                image=patch,
                mask="raisedCos",
                units="pix",
                pos=(round(get_flat_position(e, mon)) * flip, 0.0),
                size=(diameter_px - 1, diameter_px - 1),
                ori=0.0,
                color=(1, 1, 1),
                colorSpace="rgb",
                contrast=1.0,
                opacity=1.0,
                depth=0,
                interpolate=True,
                flipHoriz=False,
                flipVert=False,
                name=None,
                autoLog=False,
                maskParams=None,
            )
            stim.draw()
            for f in fixation_target:
                f.draw()
            message = visual.TextStim(
                win,
                pos=[0, -2.0],
                text="Close your %s eye and check." % side,
                height=1,
            )
            message.draw()
            win.flip()
            event.waitKeys()


def draw_help_message(win: visual.Window):
    help_message = """
    h - this screen

    p - pause the frame update

    n - show the next frame, use with (p)ause

    c - save the front buffer to disk in png format

    d - dump all frames of the current stimulus to the disk

    s - record the response of the current trial as False

    q - abort the experiment
    """
    visual.TextStim(
        win, text=dedent(help_message), pos=(0.0, 0.0), alignText="left"
    ).draw()


def main(args=None):
    params_filename = "lastParams.psydat"
    core.checkPygletDuringWait = False
    num_frames = 25
    if args is None:
        results_dir = "results"
        max_trial_count = 25
    else:
        results_dir = args.results_dir
        if args.stop_crit > 0:
            max_trial_count = args.stop_crit
        else:
            max_trial_count = -1
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    print(f"running main()...")
    
    # Start NEST server
    server_NEST = subprocess.Popen([sys.executable, \
        './NEST/NEST_Server.py', \
        './Results/server_config.json'])

    # get unfinished experiment sessions
    unfinished_experiments = ["None"]
    for _, _, files in os.walk(results_dir):
        # Manipulate files list to remove different files of same experiment
        # and present a single experiment
        print(f"files before operation: {files}")
        for i, name in enumerate(files):
            files[i] =  files[i].replace('_train_in.npy', '')
            files[i] =  files[i].replace('_train_out.npy', '')
            files[i] =  files[i].replace('_Fisher_energy.npy', '')
            files[i] =  files[i].replace('_neural_net.pth', '')
            
        print(f"files after operation: {files}")
            
        files = list(set(files))
        for name in files:
            if "_finished" not in name:
                unfinished_experiments.append(name)

    expinfo = {
        "observer":      "",
        "info":          "",
        "session":       "",
        "display":       "",
        "continue_from": "",
    }
    # check if a previous parameters file exists
    if os.path.exists(params_filename):
        expinfo = fromFile(params_filename)
        # if expinfo_loaded.keys() == expinfo.keys():
        #     expinfo = expinfo_loaded  # load only if all keys are present

    expinfo["dateStr"] = data.getDateStr()  # add the current time
    expinfo["display"] = list(supported_displays.keys())
    expinfo["continue_from"] = unfinished_experiments
    # present a dialogue to change params

    dlg = gui.DlgFromDict(
        expinfo,
        title="Temporal CSF Experiment",
        fixed=["dateStr"],
        order=["observer", "session", "info", "dateStr", "display", "continue_from"],
    )

    if dlg.OK:
        # save params to file for next time
        toFile(params_filename, expinfo)
    else:
        core.quit()  # the user hit cancel so exit
    display = get_display_params(expinfo["display"])

    mon = monitors.Monitor(
        "testMonitor"
    )  # fetch the most recent calib for this monitor
    # mon = monitors.Monitor(
    #     "acer"
    # )  # fetch the most recent calib for this monitor
    calibs = mon.calibs
    print("Calibration profiles for this display:")
    for k in calibs.keys():
        print(f"\t{k}")
    print(f"Using calibration profile: {list(calibs.keys())[-1]}")
    mon.setCurrent(list(calibs.keys())[-1])  # use the most recent calibration
    resolution = (display["PxHoriz"], display["PxVert"])
    display_fps = display["RefreshRate"]

    # mon.setSizePix(resolution)
    mon.setDistance(display["ViewingDist"] * 100)
    display_peak_cpd = 1 / pix2deg(1, mon) / 2
    print(
        "Highest spatial frequency supported by the display: %.02f cpd"
        % display_peak_cpd
    )
    patch_px = [num_frames, int(deg2pix(2, mon)), int(deg2pix(2, mon))]
    if patch_px[1] % 2 == 0:
        patch_px[1] -= 1
    if patch_px[2] % 2 == 0:
        patch_px[2] -= 1
    print("Patch size for 2 visual degrees: %dx%d" % (patch_px[1], patch_px[2]))
    
    # Define lower and upper bounds for variables. Order: f_v, f_h, f_t,
    # ecc, contrast
    print(f"patch_px: {patch_px}")
    lbs = [0.0, 0.0, 1.0, 5.0, -9]#1/512]
    ubs = [math.floor(patch_px[2]/2), math.floor(patch_px[2]/2), \
			24, 40.0, -1]#0.5]
			
    # Initialise NEST parameters
    num_dims = 5
    asymptote = 0.5
    lapse = 0.01
    w = [256, 128, 32]
	
    SNR = 5.0
    random_base = [0.0, 9e-4, 7e-4, 6e-4, 5e-4, 4e-4]
    conv_level = SNR * random_base[num_dims-1]
    
    value_dict = {'p': 0.1, 'lbs': lbs, 'ubs': ubs, 'asymptote': asymptote, \
                    'lapse': lapse, 'hidden_layers': w, 'num_dims': num_dims, \
                    'convergence_level': conv_level, 'a': 4.8, 'b': 5.8, 'c': 6.6, 'd': 5.0}

    if expinfo["continue_from"] == "None":
        savename = expinfo["observer"] + "_" + expinfo["dateStr"] + "_" + expinfo["session"] + "_"
        filename = os.path.join(
            os.path.abspath(results_dir),
            expinfo["observer"] + "_" + expinfo["dateStr"] + "_" + expinfo["session"],
        )
    else:
        savename = expinfo["continue_from"] + "_"
        filename = os.path.join(
            os.path.abspath(results_dir),
            expinfo["continue_from"],
        )
    
    # create a window
    win = visual.Window(
        size=resolution,
        fullscr=True,
        allowGUI=False,
        color=(0 - 1 / 255, 0 - 1 / 255, 0 - 1 / 255),  # background color
        monitor=mon,
        units="deg",
        waitBlanking=True,
        checkTiming=False,
        allowStencil=True,
        #multisample=False,
        numSamples=1,
        useFBO=False,
        stereo=False,
        bpc=8,  # gamma correction is not applicable for bpc>=10 ?
        winType="pyglet",
    )
    win.recordFrameIntervals = False
    if win._haveShaders:
        print("Window has shaders.")
    else:
        print("Window DOES NOT have shaders.")
    # win.saveFrameIntervals(fileName="frameintervals.txt", clear=True)
    msf = win.getMsPerFrame(nFrames=240, showVisual=True, msg="", msDelay=500.0)
    print("Measured duration per frame: %s (avg, SD, median) of duration" % str(msf))
    measured_fps = round(1 / msf[2] * 1000)
    print(f"Window FPS reported as: {measured_fps}, target is: {display_fps}")
    max_fps_deviation_tol = 1
    text_height = 1  # text height in degrees
    if abs(measured_fps - display_fps) > max_fps_deviation_tol:
        msg = f"Difference is more than {max_fps_deviation_tol} between measured and target FPS. Would you like to " \
              f"continue? [y/n] "
        message = visual.TextStim(
            win,
            pos=[0, -2.0],
            text=msg,
            height=text_height,
        )
        message.draw()
        win.flip()
        proceed = False
        while not proceed:
            keylist = event.waitKeys()
            for k in keylist:
                if k == "y":
                    proceed = True
                    message = visual.TextStim(
                        win,
                        pos=[0, -2.0],
                        text="Proceeding...",
                        height=text_height,
                    )
                    message.draw()
                    win.flip()
                elif k == "n":
                    print(
                        f"Difference is more than {max_fps_deviation_tol} between measured and target FPS."
                    )
                    print(f"Quitting...")
                    win.close()
                    core.quit()
            event.clearEvents()  # clear other (eg mouse) events - they clog the buffer

    
    fixation_target_size_px = int(deg2pix(0.6, mon)) + 1
    if fixation_target_size_px < 18:
        # minimum pixel size of the fixation target
        fixation_target_size_px = 18
    else:
        fixation_target_size_px = fixation_target_size_px // 6 * 6
    center = fixation_target(fixation_target_size_px, win)

    # check for the blind spot
    # test_blind_spot(win, center, patch_px, (10, 25, 40))

    # display instructions and wait
    start_message = "Hit a key when ready."
    if expinfo["continue_from"] != "None":
        start_message += f"\n\nResumed from experiment: {expinfo['continue_from']}"
    message1 = visual.TextStim(
        win, pos=[0, -2.0], text=start_message, alignText="center", height=text_height
    )
    message2 = visual.TextStim(
        win,
        pos=[0, -2.0],
        text="Resting Mode. Hit spacebar to resume.",
        height=text_height,
    )

    message1.draw()
    win.flip()
    event.waitKeys()
    start_t = time.time()

    # vid_idct = np.zeros(patch_px)

    # frame_idx = 0
    resting_mode = False
    trial_idx = 0
    pause_update = False
    skip_frame = False
    # mc = clock.MonotonicClock(start_time=0)
    # abort_trial = False
    # t = 0
    # event_pool = 0  # intervals (in terms of number of frames) to check for keyboard events
    
    # Connect to NEST port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('127.0.0.1', 3000))
    
    # Set config values on server
    config_dict = {'savename': savename, 'save_dir': results_dir}
    data_dict = {'message': 'SET_CONFIG', 'value': {'config_dict': config_dict}}
    config_string = json.dumps(data_dict)
    sock.send(config_string.encode(encoding='utf-8'))
    
    return_data = sock.recv(4096)
    
    # Start NEST procedure
    if expinfo["continue_from"] == "None":
        data_dict = {'message': 'INITIALISE', 'value': value_dict}
        data_string = json.dumps(data_dict)
    else:
        data_dict = {"message": "RESTART", 'value': {'data_dir': \
            os.path.abspath(results_dir), 'config_dict': value_dict}}
        data_string = json.dumps(data_dict)
    
    finished = False
    if expinfo["continue_from"] == "None":
        curr_trial = 1
    else: # Restart from middle, so read trial number from data
        train_out = np.load(filename + '_train_out.npy')
        curr_trial = train_out.shape[0] + 1
    
    
    # Show the grid of all stimuli
    zxy_set = []
    # approximately 0, 4.4, 8.8 cpds
    x_ = (0, patch_px[1] // 4, patch_px[1] // 2)
    y_ = (0, patch_px[2] // 4, patch_px[2] // 2)
    # z_ = (4, 6, 12)  # 20, 30, 60 Hz
    z_ = (1, 2, 4, 8, 12, 24)  # 2.5, 5, 10, 20, 30, 60Hz
    for z in z_:
        for x in x_:
            for y in y_:
                zxy_set.append((z, x, y))
    visualize_stimuli(patch_px, zxy_set, 0.5, win, len(z_), 9)
    
    
    while not finished:
        print(f"Trial {curr_trial}")
   
        # Request next step from NEST server
        sock.send(data_string.encode(encoding='utf-8'))

        # Get value from NEST server
        return_data = sock.recv(4096)
        return_data = json.loads(return_data.decode(encoding='utf-8')) 
        sample = return_data['next_trial']
        sample = [int(val) for val in sample[0:-1]] + [sample[-1]]
        print(f"sample: {sample}")
        
        ecc = sample[3]
        x = sample[0]
        y = sample[1]
        z = 0#sample[2]
        current_threshold = 2**sample[4]
        
        positions = ["left", "right"]
        pos_mult = {"left": -1, "right": 1}
        random.shuffle(positions)
        position = positions[0]

        show_txt = (
                "Testing with eccentricity: %d x: %d y: %d z: %d threshold: %g pos: %s"
                % (ecc, x, y, z, current_threshold, position)
        )
        print(show_txt)

        vid_idct = generate_stimulus(patch_px, x, y, z, current_threshold)
        # datestr = data.getDateStr(format='%Y-%m-%d-%H%M%S')
        # for (i, frame) in enumerate(vid_idct):
        #     im = Image.fromarray(frame*255)
        #     im = im.convert('RGB')
        #     im.save(f"avg_stim_{datestr}_{i:03d}.png", format="PNG")
        stims = []

        vid_idct = vid_idct[:, 0: patch_px[1] - 1, 0: patch_px[2] - 1]
        vid_idct = (np.round(vid_idct * 254) / 255 - 0.5) * 2

        angle = np.radians(0)  # angle to avoid the blind spot

        def getx(e):
            return e * np.cos(angle)

        def gety(e):
            return e * np.sin(angle)

        pix_dist = get_flat_position(ecc, mon)
        
        win.getMovieFrame()
        win.saveMovieFrames("./initial_frame.png")

        for frame in vid_idct:
            stims.append(
                visual.ImageStim(
                    win,
                    image=frame,
                    mask="raisedCos",
                    units="pix",
                    pos=(
                        round(getx(pix_dist) * pos_mult[position]),
                        round(gety(pix_dist)),
                    ),  # pos=(round(deg2pix(ecc, mon) * pos_mult[position]), 0.0),
                    size=(patch_px[1] - 1, patch_px[2] - 1),
                    ori=0.0,
                    color=(1, 1, 1),
                    colorSpace="rgb",
                    contrast=1.0,
                    opacity=1.0,
                    depth=0,
                    interpolate=True,
                    flipHoriz=False,
                    # texRes=128,  # Power-of-two int. Sets the resolution of the mask and texture. texRes is overridden if an array or image is provided as mask.
                    flipVert=False,
                    name=None,
                    autoLog=False,
                    maskParams=None,
                )
            )
        num_warmup_frames = display_fps // (num_frames * 2 - 1) * (num_frames * 2 - 1)
        warmup_frames = []
        frame_idx = 0
        direction = 1  # backward/forward playing direction of DCT pattern
        i = 0
        while i < num_warmup_frames:
            warmup_frames.append(
                visual.ImageStim(
                    win,
                    image=vid_idct[frame_idx] * i / num_warmup_frames,
                    mask="raisedCos",
                    units="pix",
                    pos=(
                        round(getx(pix_dist) * pos_mult[position]),
                        round(gety(pix_dist)),
                    ),  # pos=(round(deg2pix(ecc, mon) * pos_mult[position]), 0.0),
                    size=(patch_px[1] - 1, patch_px[2] - 1),
                    ori=0.0,
                    color=(1, 1, 1),
                    colorSpace="rgb",
                    contrast=1.0,
                    opacity=1.0,
                    depth=0,
                    interpolate=True,
                    flipHoriz=False,
                    # texRes=128,  # Power-of-two int. Sets the resolution of the mask and texture. texRes is overridden if an array or image is provided as mask.
                    flipVert=False,
                    name=None,
                    autoLog=False,
                    maskParams=None,
                )
            )
            frame_idx += direction
            if frame_idx == len(vid_idct) - 1:
                direction = -1
            elif frame_idx == 0:
                direction = 1
            i += 1
        resp = None
        
       

        trial_idx += 1
        if trial_idx >= 10:
            reset_display(win)
            trial_idx = 0
           
        has_converged = return_data['converged']
        if has_converged:
            print(f"Finished experiment by Fisher energy convergence.")
            finished = True
            data_dict = {"message": "TERMINATE", "value": {"finished": True}}
            data_string = json.dumps(data_dict)
            sock.send(data_string.encode(encoding='utf-8'))
        
            
        if max_trial_count != -1 and curr_trial >= max_trial_count:
            print(f"Finished experiment by number of trials.")
            finished = True
            data_dict = {"message": "TERMINATE", "value": {"finished": True}}
            data_string = json.dumps(data_dict)
            sock.send(data_string.encode(encoding='utf-8'))
        '''
        counter = visual.TextStim(
            win,
            pos=[0, -2.0],
            text="%d/%d"
                 % (
                     len(conditions) - len(stairs.runningStaircases),
                     len(conditions),
                 ),
            height=text_height,
        )
        '''
        frame_idx, direction = 0, 1
        warmup_idx = 0
        while True:
            keylist = event.getKeys()
            if keylist:
                for key in keylist:
                    if not resting_mode:
                        if (
                                key in ["left", "right", "s"]
                                and warmup_idx >= num_warmup_frames
                        ):
                            win.clearBuffer(color=True, depth=False, stencil=False)
                            win.flip()  # hide the stimulus before processing
                            if key == position:
                                resp = 1  # correct
                            else:
                                resp = 0  # incorrect
                            # reset the temporal position
                            frame_idx = 0
                            direction = 1
                        elif key == "q":
                            # abort the experiment
                            win.close()
                            # plt.plot(win.frameIntervals)
                            # plt.show()
                            core.quit()
                        elif key == "p":
                            pause_update = not pause_update
                        elif key == "n":
                            skip_frame = not skip_frame
                        elif key == "c":
                            # save the current frame to disk
                            buffer = win.getMovieFrame(buffer="front")
                            buffer.save(
                                f"screenshot_{data.getDateStr(format='%Y-%m-%d-%H%M%S')}.png",
                                "PNG",
                            )
                        elif key == "d":
                            # dump the stimuli to disk for analysis
                            for frame_idx in range(len(stims)):
                                stims[frame_idx].draw()
                                win.flip()
                                buffer = win.getMovieFrame(buffer="front")
                                buffer.save(f"frame_{frame_idx:03d}.png", "PNG")
                        elif key == "h":
                            # display help
                            win.flip()
                            draw_help_message(win)
                            win.flip()
                            event.waitKeys()
                            win.flip()
                    if key == "space":
                        resting_mode = not resting_mode
                        if not resting_mode:
                            reset_display(win)
            event.clearEvents()  # clear other (eg mouse) events - they clog the buffer

            if resting_mode:
                message2.draw()
                for s in center:
                    s.draw()
                win.flip()
            else:
                if resp is not None:
                    # Create data to send to NEST, which asks for new trial
                    print(f"Response value: {resp}")
                    data_dict = {'message': 'NEXT_TRIAL', 'value': resp}
                    data_string = json.dumps(data_dict)
                    #stairs.addResponse(resp)
                    break

                if warmup_idx < num_warmup_frames:
                    warmup_frames[warmup_idx].draw()
                    for s in center:
                        s.draw()
                    #counter.draw()
                    win.flip()
                    
                    if warmup_idx == 0:
                        win.getMovieFrame()
                        win.saveMovieFrames("./initial_frame.png")
                    
                    warmup_idx += 1
                else:
                    # No response yet, prepare the next frame
                    if not pause_update:
                        frame_idx += direction
                    elif skip_frame:
                        frame_idx += direction
                        skip_frame = not skip_frame

                    if frame_idx == len(vid_idct) - 1:
                        direction = -1
                    elif frame_idx == 0:
                        direction = 1

                    #counter.draw()
                    for s in center:
                        s.draw()
                    stims[frame_idx].draw()
                    win.flip()
                    
                    win.getMovieFrame()
                    win.saveMovieFrames("./stimulus.png")
        
        curr_trial += 1

    '''
    stairs.saveAsExcel(filename)  # easy to browse
    stairs.saveAsPickle(filename)  # contains more info
    try:
        os.remove(filename + "_unfinished.psydat")  # remove the checkpoints
        os.remove(filename + "_conditions.psydat")
    except FileNotFoundError:
        pass
    '''
    session_dur = time.time() - start_t
    session_min = int(session_dur // 60)
    print(f"The session took {session_min}m {session_dur - (session_min * 60):.02f}s.")
    win.close()
    core.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Path to the results directory",
    )
    parser.add_argument(
		"--stop-crit",
		type=int,
		default=25,
		help="Stopping criterion. Fixed number of samples if positive, otherwise using NEST convergence.",
	)
    args = parser.parse_args()
    main(args)
