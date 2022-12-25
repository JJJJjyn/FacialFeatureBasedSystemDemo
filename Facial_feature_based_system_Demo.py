import wx
import wx.xrc
import wx.adv
from threading import Thread
import dlib
import numpy as np
import cv2
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import datetime,time
import math
from sklearn.preprocessing import StandardScaler
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

COVER = 'load.png'

class Fatigue_detection(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__ (self, parent, id = wx.ID_ANY, title = u"Fatigue System Demo - base on facial featue", 
                           pos = wx.DefaultPosition, size = wx.Size(1239,810), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL)
        
        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)
        self.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNTEXT))
        self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_APPWORKSPACE))
        
        bSizer1 = wx.BoxSizer(wx.VERTICAL)
        
        bSizer2 = wx.BoxSizer(wx.HORIZONTAL)
        
        bSizer3 = wx.BoxSizer(wx.VERTICAL)
        
        self.m_animCtrl1 = wx.adv.AnimationCtrl(self, wx.ID_ANY, wx.adv.NullAnimation, wx.Point( -1,-1 ), wx.Size( -1,-1 ), 
                                                wx.adv.AC_NO_AUTORESIZE) 
        self.m_animCtrl1.SetInactiveBitmap(wx.NullBitmap)
        bSizer3.Add(self.m_animCtrl1, 1, wx.EXPAND|wx.ALL, 5)
        
        bSizer2.Add(bSizer3, 9, wx.ALL|wx.EXPAND, 5)
        
        bSizer4 = wx.BoxSizer(wx.VERTICAL)
        
        sbSizer1 = wx.StaticBoxSizer(wx.StaticBox( self, wx.ID_ANY, u"Operations" ), wx.VERTICAL)
        
        sbSizer2 = wx.StaticBoxSizer(wx.StaticBox( sbSizer1.GetStaticBox(), wx.ID_ANY, u"Video source"), wx.VERTICAL)
        
        gSizer1 = wx.GridSizer(0, 2, 0, 8)
        
        self.m_button5 = wx.Button(sbSizer2.GetStaticBox(), wx.ID_ANY, u"Real-time camera", wx.DefaultPosition, wx.Size(130,24), 0)
        gSizer1.Add(self.m_button5, 0, wx.ALL, 5)
        
        self.m_button4 = wx.Button(sbSizer2.GetStaticBox(), wx.ID_ANY, u"Upload video", wx.DefaultPosition, wx.Size(130,24), 0)
        gSizer1.Add(self.m_button4, 0, wx.ALL, 5)
        
        sbSizer2.Add(gSizer1, 0, wx.ALL|wx.EXPAND, 5)
        
        sbSizer1.Add(sbSizer2, 0, wx.EXPAND|wx.ALL, 5)
        
        sbSizer3 = wx.StaticBoxSizer(wx.StaticBox(sbSizer1.GetStaticBox(), wx.ID_ANY, u"Facial features"), wx.VERTICAL)
        
        bSizer5 = wx.BoxSizer(wx.HORIZONTAL)
        
        self.eyes_checkBox1 = wx.CheckBox(sbSizer3.GetStaticBox(), wx.ID_ANY, u"   Eyes-tracking", wx.DefaultPosition, wx.Size(130,19), 0)
        self.eyes_checkBox1.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_CAPTIONTEXT))
        self.eyes_checkBox1.SetBackgroundColour( wx.SystemSettings.GetColour(wx.SYS_COLOUR_APPWORKSPACE))
        
        bSizer5.Add(self.eyes_checkBox1, 0, wx.ALL, 5)
        
        self.yawn_checkBox2 = wx.CheckBox(sbSizer3.GetStaticBox(), wx.ID_ANY, u"         Yawn", wx.DefaultPosition, wx.Size(130,19), 0)
        self.yawn_checkBox2.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_CAPTIONTEXT))
        self.yawn_checkBox2.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_APPWORKSPACE))
        
        bSizer5.Add(self.yawn_checkBox2, 0, wx.ALL, 5)
        
        sbSizer3.Add(bSizer5, 0, wx.EXPAND|wx.ALL, 5)
        
        bSizer6 = wx.BoxSizer(wx.HORIZONTAL)
        
        self.blink_checkBox4 = wx.CheckBox(sbSizer3.GetStaticBox(), wx.ID_ANY, u"      Blink", wx.DefaultPosition, wx.Size(130,19), 0)
        self.blink_checkBox4.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_CAPTIONTEXT))
        self.blink_checkBox4.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_APPWORKSPACE))
        
        bSizer6.Add(self.blink_checkBox4, 0, wx.ALL, 5)
        
        self.nod_checkBox5 = wx.CheckBox(sbSizer3.GetStaticBox(), wx.ID_ANY, u"         Nod", wx.DefaultPosition, wx.Size(130,19), 0)
        self.nod_checkBox5.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_CAPTIONTEXT))
        self.nod_checkBox5.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_APPWORKSPACE))
        
        bSizer6.Add(self.nod_checkBox5, 0, wx.ALL, 5)
        
        sbSizer3.Add(bSizer6, 0, wx.EXPAND|wx.ALL, 5)
        
        sbSizer1.Add(sbSizer3, 0, wx.EXPAND|wx.ALL, 5)
        
        sbSizer4 = wx.StaticBoxSizer(wx.StaticBox( sbSizer1.GetStaticBox(), wx.ID_ANY, u"Select task" ), wx.VERTICAL)
        
        bSizer8 = wx.BoxSizer(wx.HORIZONTAL)
        
        self.m_radioBtn1 = wx.RadioButton(sbSizer4.GetStaticBox(), wx.ID_ANY, u" Fatigue detection", wx.DefaultPosition, wx.Size(135,19), 0)
        bSizer8.Add(self.m_radioBtn1, 0, wx.ALL, 5)
        
        self.m_radioBtn2 = wx.RadioButton(sbSizer4.GetStaticBox(), wx.ID_ANY, u" Feature detection", wx.DefaultPosition, wx.Size( 135,19 ), 0)
        self.m_radioBtn2.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_CAPTIONTEXT))
        self.m_radioBtn2.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_APPWORKSPACE))
        
        bSizer8.Add(self.m_radioBtn2, 0, wx.ALL, 5)
        
        sbSizer4.Add(bSizer8, 0, wx.EXPAND|wx.ALL, 5)
        
        sbSizer1.Add(sbSizer4, 0, wx.EXPAND|wx.ALL, 5)
        
        sbSizer6 = wx.StaticBoxSizer(wx.StaticBox(sbSizer1.GetStaticBox(), wx.ID_ANY, u"Thresholds setting"), wx.VERTICAL)
        
        bSizer101 = wx.BoxSizer(wx.HORIZONTAL)
        
        self.m_staticText1 = wx.StaticText(sbSizer6.GetStaticBox(), wx.ID_ANY, u"f_thre: ", wx.DefaultPosition, wx.Size(40,19), 0)
        self.m_staticText1.Wrap(-1)
        bSizer101.Add(self.m_staticText1, 0, wx.TOP|wx.BOTTOM|wx.LEFT, 5)
        
        m_listBox1Choices = [u"1",u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"10", u"11", u"12", u"13", 
                             u"14", u"15", u"16", u"17", u"18"]
        self.m_listBox1 = wx.ListBox(sbSizer6.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size(48,24), m_listBox1Choices, 0)
        bSizer101.Add(self.m_listBox1, 0, wx.TOP|wx.BOTTOM|wx.RIGHT, 5)
        
        self.m_staticText2 = wx.StaticText(sbSizer6.GetStaticBox(), wx.ID_ANY, u"ear_thre: ", wx.DefaultPosition, wx.Size(53,19), 0)
        self.m_staticText2.Wrap(-1)
        bSizer101.Add(self.m_staticText2, 0, wx.TOP|wx.BOTTOM|wx.LEFT, 5)
        
        m_listBox2Choices = [u"0.12", u"0.14",u"0.16", u"0.18", u"0.2", u"0.22", u"0.24", u"0.26", u"0.28", u"0.30", u"0.32"]
        self.m_listBox2 = wx.ListBox(sbSizer6.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size(50,24), m_listBox2Choices, 0)
        bSizer101.Add(self.m_listBox2, 0, wx.TOP|wx.BOTTOM|wx.RIGHT, 5)
        
        self.m_staticText3 = wx.StaticText(sbSizer6.GetStaticBox(), wx.ID_ANY, u"bf_thre:  ", wx.DefaultPosition, wx.Size(45,19), 0)
        self.m_staticText3.Wrap(-1)
        bSizer101.Add(self.m_staticText3, 0, wx.TOP|wx.BOTTOM|wx.LEFT, 5)
        
        m_listBox3Choices = [u"1",u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"10", u"11", u"12", u"13", 
                             u"14", u"15", u"16", u"17", u"18"]
        self.m_listBox3 = wx.ListBox(sbSizer6.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size(38,24), m_listBox3Choices, 0)
        bSizer101.Add(self.m_listBox3, 0, wx.TOP|wx.BOTTOM|wx.RIGHT, 5)
        
        sbSizer6.Add(bSizer101, 0, wx.EXPAND, 5)
        
        bSizer11 = wx.BoxSizer(wx.HORIZONTAL)
        
        self.m_staticText4 = wx.StaticText(sbSizer6.GetStaticBox(), wx.ID_ANY, u"mar_thre:", wx.DefaultPosition, wx.Size(53,19), 0)
        self.m_staticText4.Wrap(-1)
        bSizer11.Add(self.m_staticText4, 0, wx.TOP|wx.BOTTOM|wx.LEFT, 5)
        
        m_listBox4Choices = [u"0.1", u"0.15",u"0.2", u"0.25",u"0.3", u"0.35", u"0.4", u"0.45", u"0.5",u"0.55", u"0.6"]
        self.m_listBox4 = wx.ListBox(sbSizer6.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size(50,24), m_listBox4Choices, 0)
        bSizer11.Add(self.m_listBox4, 0, wx.ALL, 5)
        
        self.m_staticText5 = wx.StaticText(sbSizer6.GetStaticBox(), wx.ID_ANY, u"yf_thre:", wx.DefaultPosition, wx.Size(45,19), 0)
        self.m_staticText5.Wrap(-1)
        bSizer11.Add(self.m_staticText5, 0, wx.TOP|wx.BOTTOM|wx.LEFT, 5)
        
        m_listBox5Choices = [u"1",u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"10", u"11", u"12", u"13", 
                             u"14", u"15", u"16", u"17", u"18"]
        self.m_listBox5 = wx.ListBox(sbSizer6.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size(38,24), m_listBox5Choices, 0)
        bSizer11.Add(self.m_listBox5, 0, wx.TOP|wx.BOTTOM|wx.RIGHT, 5)
        
        sbSizer6.Add(bSizer11, 0, wx.EXPAND, 5)
        
        bSizer12 = wx.BoxSizer(wx.HORIZONTAL)
        
        self.m_staticText6 = wx.StaticText(sbSizer6.GetStaticBox(), wx.ID_ANY, u"nod_thre:", wx.DefaultPosition, wx.Size(57,19), 0)
        self.m_staticText6.Wrap(-1)
        bSizer12.Add(self.m_staticText6, 0, wx.TOP|wx.BOTTOM|wx.LEFT, 5)
                
        m_listBox6Choices = [u"3", u"5", u"8", u"10", u"12", u"15", u"20", u"25", u"30", u"35", u"40"]
        self.m_listBox6 = wx.ListBox(sbSizer6.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size(50,24), m_listBox6Choices, 0)
        bSizer12.Add(self.m_listBox6, 0, wx.TOP|wx.BOTTOM|wx.RIGHT, 5)
        
        self.m_staticText8 = wx.StaticText(sbSizer6.GetStaticBox(), wx.ID_ANY, u"nf_thre:", wx.DefaultPosition, wx.Size(47,19), 
                                           wx.ALIGN_RIGHT)
        self.m_staticText8.Wrap(-1)
        bSizer12.Add(self.m_staticText8, 0, wx.ALL, 5)
        
        m_listBox8Choices = [u"1",u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"10", u"11", u"12", u"13", 
                             u"14", u"15", u"16", u"17", u"18"]
        self.m_listBox8 = wx.ListBox(sbSizer6.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size(38,24), m_listBox8Choices, 0)
        bSizer12.Add(self.m_listBox8, 0, wx.TOP|wx.BOTTOM|wx.RIGHT, 5)
        
        sbSizer6.Add(bSizer12, 0, wx.EXPAND, 5)
        
        sbSizer1.Add(sbSizer6, 0, wx.ALL|wx.EXPAND, 5)
        
        bSizer9 = wx.BoxSizer(wx.VERTICAL)
        
        bSizer10 = wx.BoxSizer(wx.HORIZONTAL)
        
        self.on_button1 = wx.Button(sbSizer1.GetStaticBox(), wx.ID_ANY, u"Start detection", wx.DefaultPosition, wx.Size(140,24), 0)
        self.on_button1.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNTEXT))
        self.on_button1.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DLIGHT))
        
        bSizer10.Add(self.on_button1, 0, wx.ALL, 5)
        
        self.off_button3 = wx.Button(sbSizer1.GetStaticBox(), wx.ID_ANY, u"Stop detection", wx.DefaultPosition, wx.Size(140,24), 0)
        bSizer10.Add(self.off_button3, 0, wx.ALL, 5)
        
        bSizer9.Add(bSizer10, 0, wx.ALL|wx.EXPAND, 5)
        
        sbSizer1.Add(bSizer9, 0, wx.ALL|wx.EXPAND, 5)
        
        sbSizer5 = wx.StaticBoxSizer(wx.StaticBox(sbSizer1.GetStaticBox(), wx.ID_ANY, u"Outputs"), wx.VERTICAL)
        
        self.m_textCtrl3 = wx.TextCtrl(sbSizer5.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size(-1,-1), 
                                       wx.TE_MULTILINE|wx.TE_READONLY)
        self.m_textCtrl3.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNTEXT))
        
        sbSizer5.Add(self.m_textCtrl3, 1, wx.ALL|wx.EXPAND, 5)
        
        sbSizer1.Add(sbSizer5, 1, wx.EXPAND|wx.ALL, 5)
        
        bSizer4.Add(sbSizer1, 1, wx.EXPAND|wx.ALL, 5)
        
        bSizer2.Add(bSizer4, 3, wx.EXPAND|wx.ALL, 5)
        
        bSizer1.Add(bSizer2, 1, wx.ALL|wx.EXPAND, 5)
        
        self.SetSizer(bSizer1)
        self.Layout()
        
        self.Centre(wx.BOTH)
        
        self.image_cover = wx.Image(COVER, wx.BITMAP_TYPE_ANY)
        self.bmp = wx.StaticBitmap(self.m_animCtrl1, -1, wx.Bitmap(self.image_cover))
        
        self.icon = wx.Icon('1.ico', wx.BITMAP_TYPE_ICO)
        self.SetIcon(self.icon)
        
        # Connect Events
        self.m_button5.Bind(wx.EVT_BUTTON, self.camera_on)
        self.m_button4.Bind(wx.EVT_BUTTON, self.upload_video)
        self.m_listBox1.Bind(wx.EVT_LISTBOX, self.f_thre)
        self.m_listBox2.Bind(wx.EVT_LISTBOX, self.ear_thre)
        self.m_listBox3.Bind(wx.EVT_LISTBOX, self.bf_thre)
        self.m_listBox4.Bind(wx.EVT_LISTBOX, self.mar_thre)
        self.m_listBox5.Bind(wx.EVT_LISTBOX, self.yf_thre)
        self.m_listBox6.Bind(wx.EVT_LISTBOX, self.nod_thre)
        self.m_listBox8.Bind(wx.EVT_LISTBOX, self.nf_thre )
        self.on_button1.Bind(wx.EVT_BUTTON, self.on)
        self.off_button3.Bind(wx.EVT_BUTTON, self.off)
        
        
        #parameters
        #default
        # real-time camera
        self.VIDEO_STREAM = 0
        self.CAMERA_STYLE = False
        
        # Flicker Threshold (seconds)
        self.AR_CONSEC_FRAMES_check = 3
        self.OUT_AR_CONSEC_FRAMES_check = 5 #no face detected
        
        self.FIX_THRESH = 3
        
        self.EYE_AR_THRESH = 0.24
        self.EYE_AR_CONSEC_FRAMES = 8
        
        self.MAR_THRESH = 0.35
        self.MOUTH_AR_CONSEC_FRAMES = 8
        
        #self.HAR_THRESH = 0.3
        self.NOD_AR_CONSEC_FRAMES = 8
        
        self.pTHRE = 15
        self.rTHRE = 15
        self.yTHRE =15
        
        #initialization
        # fixation
        self.FIX = 0
        # frame counter and blink total
        self.COUNTER = 0
        self.TOTAL = 0
        # frame counter and yawn total
        self.mCOUNTER = 0
        self.mTOTAL = 0
        # frame counter and nod total
        self.hCOUNTER = 0
        self.hTOTAL = 0
        self.PUPIL = []
        
        #3D reference point, the model from http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],  
                                 [1.330353, 7.122144, 6.903745],  
                                 [-1.330353, 7.122144, 6.903745], 
                                 [-6.825897, 6.760612, 4.402142], 
                                 [5.311432, 5.485328, 3.987654],  
                                 [1.789930, 5.393625, 4.413414],  
                                 [-1.789930, 5.393625, 4.413414], 
                                 [-5.311432, 5.485328, 3.987654],
                                 [2.005628, 1.409845, 6.165652],
                                 [-2.005628, 1.409845, 6.165652],
                                 [2.774015, -2.080775, 5.048531], 
                                 [-2.774015, -2.080775, 5.048531],
                                 [0.000000, -3.116408, 6.097667],
                                 [0.000000, -7.415691, 4.070434]])
        
        # camera coordinate system(xyz) 3D
        self.K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
                 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
                 0.0, 0.0, 1.0]
        # camera distortion parameters
        self.D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

        # Pixel coordinate system (xy) 2D
        self.cam_matrix = np.array(self.K).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(self.D).reshape(5, 1).astype(np.float32)

        # reproject 3D point coordinate axes to verify resulting pose
        self.reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                       [10.0, 10.0, -10.0],
                                       [10.0, -10.0, -10.0],
                                       [10.0, -10.0, 10.0],
                                       [-10.0, 10.0, 10.0],
                                       [-10.0, 10.0, -10.0],
                                       [-10.0, -10.0, -10.0],
                                       [-10.0, -10.0, 10.0]])
        # cube 12 axes
        self.line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                          [4, 5], [5, 6], [6, 7], [7, 4],
                          [0, 4], [1, 5], [2, 6], [3, 7]]
        
        self.m_textCtrl3.AppendText(u"Welcome to the Demo!!!\n")

    def __del__(self):
        pass
    
    def get_head_pose(self,shape):
        # 2D reference point，according to https://ibug.doc.ic.ac.uk/resources/300-W/
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                    shape[39], shape[42], shape[45], shape[31], shape[35],
                                    shape[48], shape[54], shape[57], shape[8]])
        
        # solvePnP - compute rotation and translation matrices:
        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs)
            
        # The distance between the original 2d point and the reprojected 2d point
        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec, self.cam_matrix,self.dist_coeffs)
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))# 以8行2列显示

        # Ecalc euler angle
        # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        # decomposeProjectionMatrix
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        pitch, yaw, roll = [math.radians(_) for _ in euler_angle]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        #print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

        return reprojectdst, euler_angle, pitch, roll, yaw
    
    def eye_aspect_ratio(self,eye):
        # EAR
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)

        return ear
    
    def mouth_aspect_ratio(self,mouth):
        #MAR
        A = np.linalg.norm(mouth[13] - mouth[19])
        B = np.linalg.norm(mouth[15] - mouth[17])
        C = np.linalg.norm(mouth[12] - mouth[16])
        mar = (A + B) / (2.0 * C)
        return mar
    
    def _learning_face(self,event):
        # dlib for face detect
        self.detector = dlib.get_frontal_face_detector()
        # 68-landmarks model
        self.predictor = dlib.shape_predictor("C:/Users/jyn/shape_predictor_68_face_landmarks.dat")
        self.m_textCtrl3.AppendText(u"Successfully load 68-landmarks model!!!\n")
        # facial feature index
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
            
        # loop, read the frame
        while(self.cap.isOpened()):
            # cap.read()
            flag, im_rd = self.cap.read()
            # grayscale
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
            
            # face detect
            faces = self.detector(img_gray, 0)
            if(len(faces)!=0):
                for k, d in enumerate(faces):
                    # highlight the face
                    cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (152,251,152),1)
                    # landmarks
                    shape = self.predictor(im_rd, d)
                    # highlight landmarks
                    for i in range(68):
                        cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (50,205,50), -1, 8)
                    shape = face_utils.shape_to_np(shape)
                    
                    #eyes-tracking
                    if self.eyes_checkBox1.GetValue()== True:
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        lp = np.mean(leftEye, axis = 0)
                        lp = np.array([int(lp[0]),int(lp[1])])
                        rp = np.mean(rightEye, axis = 0)
                        rp = np.array([int(rp[0]),int(rp[1])])
                        cv2.circle(im_rd, lp, 4, (0,0,255), -1, 8)
                        cv2.circle(im_rd, rp, 4, (0,0,255), -1, 8)
                        
                        self.PUPIL.append(lp)
                        self.PUPIL.append(rp)
                        
                        if len(self.PUPIL) > 2*self.FIX_THRESH:
                            del(self.PUPIL[0],self.PUPIL[0])
                            p = np.array(self.PUPIL)
                            #trans = StandardScaler()
                            #p = trans.fit_transform(p)
                            self.FIX = round(np.std(p,ddof=1),2)
                        
                        cv2.putText(im_rd, "Fixation variance: {}".format(self.FIX), (550, 30),cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (193,182,255), 2)

                    # yawn
                    if self.yawn_checkBox2.GetValue()== True:
                        mouth = shape[mStart:mEnd]
                        innermouth = [mouth[12],mouth[13],mouth[15],mouth[16],mouth[17],mouth[19]]
                        mar = self.mouth_aspect_ratio(mouth)
                        innermouth = cv2.convexHull(np.array(innermouth))
                        # draw contour
                        cv2.drawContours(im_rd, [innermouth], -1, (50,205,50), 1)
                        # yawn detect
                        if mar > self.MAR_THRESH:
                            self.mCOUNTER += 1
                        else:
                            # Mar > self.MAR_THRESH last for not less than self.MOUTH_AR_CONSEC_FRAMES, it means a yawn
                            if self.mCOUNTER >= self.MOUTH_AR_CONSEC_FRAMES:
                                self.mTOTAL += 1
                                #display
                                ####cv2.putText(im_rd, "Yawning!", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                                self.m_textCtrl3.AppendText(time.strftime('%Y-%m-%d %H:%M ', time.localtime())+u"Yawning!\n")
                            # reset COUNTER
                            self.mCOUNTER = 0
                        
                        if self.m_radioBtn2.GetValue()== True:
                            #cv2.putText(im_rd, "COUNTER: {}".format(self.mCOUNTER), (150, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2) 
                            cv2.putText(im_rd, "MAR: {:.2f}".format(mar), (300, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        cv2.putText(im_rd, "Yawning: {}".format(self.mTOTAL), (550, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (193,182,255), 2)
                    else:
                        pass
                    
                    # blink
                    if self.blink_checkBox4.GetValue()== True:
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        # average EAR
                        leftEAR = self.eye_aspect_ratio(leftEye)
                        rightEAR = self.eye_aspect_ratio(rightEye)
                        ear = (leftEAR + rightEAR) / 2.0
                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)
                        # draw contour
                        cv2.drawContours(im_rd, [leftEyeHull], -1, (50,205,50), 1)
                        cv2.drawContours(im_rd, [rightEyeHull], -1, (50,205,50), 1)
                        # blink detect
                        if ear < self.EYE_AR_THRESH:
                            self.COUNTER += 1
                        else:
                            # average EAR < self.EYE_AR_THRESH last for not less than self.EYE_AR_CONSEC_FRAMES, it means a yawn
                            if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                                self.TOTAL += 1
                                self.m_textCtrl3.AppendText(time.strftime('%Y-%m-%d %H:%M ', time.localtime())+u"Blink!\n")
                            # reset COUNTER
                            self.COUNTER = 0
                        
                        if self.m_radioBtn2.GetValue()== True:
                            #cv2.putText(im_rd, "Faces: {}".format(len(faces)), (150, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)     
                            #cv2.putText(im_rd, "COUNTER: {}".format(self.COUNTER), (150, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2) 
                            cv2.putText(im_rd, "EAR: {:.2f}".format(ear), (300, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        cv2.putText(im_rd, "Blinks: {}".format(self.TOTAL), (550, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (193,182,255), 2)
                    else:
                        pass
                    
                    # nod
                    if self.nod_checkBox5.GetValue()== True:
                        reprojectdst, euler_angle, pitch, roll, yaw = self.get_head_pose(shape) 
                        har = euler_angle[0, 0]
                        if abs(pitch) > self.pTHRE or abs(roll) >self.rTHRE or abs(yaw) > self.yTHRE:
                            self.hCOUNTER += 1
                        else:
                            # har > self.HAR_THRESH last for not less than self.NOD_AR_CONSEC_FRAMES, it means a yawn
                            if self.hCOUNTER >= self.NOD_AR_CONSEC_FRAMES:
                                self.hTOTAL += 1
                                self.m_textCtrl3.AppendText(time.strftime('%Y-%m-%d %H:%M ', time.localtime())+u"Nod!\n")
                            # reset COUNTER
                            self.hCOUNTER = 0
                        # Draw a cube with 12 axes
                        for start, end in self.line_pairs:
                            cv2.line(im_rd, (int(reprojectdst[start][0]),int(reprojectdst[start][1])), 
                                     (int(reprojectdst[end][0]),int(reprojectdst[end][1])), (208,224,64))
                        
                        if self.m_radioBtn2.GetValue()== True:
                            ####cv2.putText(im_rd, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (150, 120), 
                                        ####cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,255), thickness=2)# GREEN
                            ####cv2.putText(im_rd, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (300, 120), 
                                        ####cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,255), thickness=2)# BLUE
                            ####cv2.putText(im_rd, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (450, 120), 
                                        ####cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,255), thickness=2)# RED   
                            cv2.putText(im_rd, "P: " + "{:7.2f}".format(pitch), (150, 120), 
                                        cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,255), thickness=2)# GREEN
                            cv2.putText(im_rd, "R: " + "{:7.2f}".format(roll), (300, 120), 
                                        cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,255), thickness=2)# BLUE
                            cv2.putText(im_rd, "Y: " + "{:7.2f}".format(yaw), (450, 120), 
                                        cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,255), thickness=2)# RED    
                        cv2.putText(im_rd, "Nod: {}".format(self.hTOTAL), (550, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (193,182,255), 2)
                    else:
                        pass
                    
                #print('MAR:{:.2f} '.format(mar)+"\tYawn ? "+str([False,True][mar > self.MAR_THRESH]))
                #print('EAR:{:.2f} '.format(ear)+"\tBlink? "+str([False,True][self.COUNTER>=1]))
                
            else:
                # fail to detect face
                cv2.putText(im_rd, "No Face! ", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (193,182,255),3, cv2.LINE_AA)
                
            # The system issues fatigue warnings
            # Blink 50 times, yawn 15 times, nod 30 times
            if self.m_radioBtn1.GetValue()== True:
                if self.TOTAL >= 30 or self.mTOTAL>= 2 or self.hTOTAL>=15:
                    cv2.putText(im_rd, "Fatigue Warning!!!", (100, 200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                    self.m_textCtrl3.AppendText(u"Notice: Fatigue Warning!!!!!!!!!!!!\n")
                
            # opencv: BGR，wxPython: RGB
            height,width = im_rd.shape[:2]
            image1 = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)
            pic = wx.Bitmap.FromBuffer(width,height,image1)
            
            self.bmp.SetBitmap(pic)

        # off camera
        self.cap.release()
    
    # Virtual event handlers
    def camera_on(self, event):
        def resizeBitmap(image, width=100, height=100):
            bmp = image.Scale(width, height).ConvertToBitmap()
            return bmp
        # turn the camera on. can choose different cameras by different self.VIDEO_STREAM
        self.VIDEO_STREAM = 0
        self.cap = cv2.VideoCapture(self.VIDEO_STREAM)
        if self.cap.isOpened()==True:
            self.CAMERA_STYLE = True
            self.m_textCtrl3.AppendText(u"Camera is ready!!!\n")
            self.m_textCtrl3.AppendText(u"Please click button to start detection!!!\n")
            self.bmp.SetBitmap(wx.Bitmap(self.image_cover))
            
        else:
            self.m_textCtrl3.AppendText(u"Error: failed to turn on the camera!!!\n")
            self.bmp.SetBitmap(wx.Bitmap(self.image_cover))
    
    def upload_video(self, event):
        if self.CAMERA_STYLE == True :
            # turn off camera
            dlg = wx.MessageDialog(None, u'Are you sure to turn off the camera？', u'Please choose', wx.YES_NO | wx.ICON_QUESTION)
            if(dlg.ShowModal() == wx.ID_YES):
                self.cap.release()
                self.CAMERA_STYLE = False
                self.bmp.SetBitmap(wx.Bitmap(self.image_cover))
                dlg.Destroy()
                
        # choose video path
        dialog = wx.FileDialog(self,u"Please choose video",os.getcwd(),'',wildcard="(*.mp4)|*.mp4",style=wx.FD_OPEN | wx.FD_CHANGE_DIR)
        if dialog.ShowModal() == wx.ID_OK:
            #load video
            self.m_textCtrl3.SetValue(u"Video path:"+dialog.GetPath()+"\n")
            self.VIDEO_STREAM = str(dialog.GetPath())
            dialog.Destroy
    
            self.m_textCtrl3.AppendText(u"Video is ready!!!\n") 
            self.m_textCtrl3.AppendText(u"Please click button to start detection!!!\n")
    
    def f_thre(self, event):
        self.m_textCtrl3.AppendText(u"Set the fixation threshold to: "+event.GetString()+u"\n")
        self.FIX_THRESH = float(event.GetString())
    
    def ear_thre(self, event):
        self.m_textCtrl3.AppendText(u"Set the EAR threshold to: "+event.GetString()+u"\n")
        self.EYE_AR_THRESH = float(event.GetString())
    
    def bf_thre(self, event):
        self.m_textCtrl3.AppendText(u"Set the blink frames threshold to: "+event.GetString()+u"\n")
        self.EYE_AR_CONSEC_FRAMES = float(event.GetString())
    
    def mar_thre(self, event):
        self.m_textCtrl3.AppendText(u"Set the MAR threshold to: "+event.GetString()+u"\n")
        self.MAR_THRESH = float(event.GetString())
    
    def yf_thre(self, event):
        self.m_textCtrl3.AppendText(u"Set the yawn frames threshold to: "+event.GetString()+u"\n")
        self.MOUTH_AR_CONSEC_FRAMES = float(event.GetString())
    
    def nod_thre(self, event):
        self.m_textCtrl3.AppendText(u"Set the nod threshold to: "+event.GetString()+u"\n")
        self.pTHRE = self.rTHRE = self.yTHRE  = float(event.GetString())
    
    def nf_thre(self, event):
        self.m_textCtrl3.AppendText(u"Set the nod frames threshold to: "+event.GetString()+u"\n")
        #########
        self.NOD_AR_CONSEC_FRAMES = float(event.GetString())
    
    def on(self, event):
        self.cap = cv2.VideoCapture(self.VIDEO_STREAM)
        self.m_textCtrl3.AppendText(u"Start detecting!!!\n")
        import _thread
        _thread.start_new_thread(self._learning_face, (event,))
    
    def off(self, event):
        self.cap.release()
        self.bmp.SetBitmap(wx.Bitmap(self.image_cover))
        if self.CAMERA_STYLE == True :
            self.CAMERA_STYLE = False
            self.m_textCtrl3.AppendText(u"Trun off the camera!!!\n")
        self.m_textCtrl3.AppendText(u"Finish the detection!!!\n")
        #reset
        self.FIX = 0
        self.COUNTER = 0
        self.TOTAL = 0
        self.mCOUNTER = 0
        self.mTOTAL = 0
        self.hCOUNTER = 0
        self.hTOTAL = 0
        self.PUPIL = []
        
class system_UI_demo1(wx.App):
    def OnInit(self):
        self.frame = Fatigue_detection(parent=None)
        self.frame.Show(True)
        return True  
