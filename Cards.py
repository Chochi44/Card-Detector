# packages
import numpy as np
import cv2
import time

# Adaptive threshold levels
BKG_THRESH = 60
CARD_THRESH = 30

# Width and height of card corner, where rank and suit are
CORNER_WIDTH = 32
CORNER_HEIGHT = 84

# Dimensions of rank train images
RANK_WIDTH = 70
RANK_HEIGHT = 125

# Dimensions of suit train images
SUIT_WIDTH = 70
SUIT_HEIGHT = 100
RANK_DIFF_MAX = 2000
SUIT_DIFF_MAX = 700

CARD_MAX_AREA = 120000
CARD_MIN_AREA = 25000

font = cv2.FONT_HERSHEY_SIMPLEX

### Structures to hold query card and train card information ###


class Query_card:

    def __init__(self):
        self.contour = []  # Contour of card
        self.width, self.height = 0, 0  # Width and height of card
        self.corner_pts = []  # Corner points of card
        self.center = []  # Center point of card
        self.warp = []  # 200x300, flattened, grayed, blurred image
        self.rank_img = []  # Thresholded, sized image of card's rank
        self.suit_img = []  # Thresholded, sized image of card's suit
        self.best_rank_match = "Unknown"  # Best matched rank
        self.best_suit_match = "Unknown"  # Best matched suit
        self.rank_diff = 0  # Difference between rank image and best matched train rank image
        self.suit_diff = 0  # Difference between suit image and best matched train suit image


class Train_ranks:

    def __init__(self):
        self.img = []
        self.name = "Placeholder"


class Train_suits:

    def __init__(self):
        self.img = []
        self.name = "Placeholder"


def load_ranks(filepath):

    train_ranks = []
    i = 0

    for Rank in ['Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven',
                 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King']:
        train_ranks.append(Train_ranks())
        train_ranks[i].name = Rank
        filename = Rank + '.jpg'
        train_ranks[i].img = cv2.imread(
            filepath+filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    return train_ranks


def load_suits(filepath):

    train_suits = []
    i = 0

    for Suit in ['Spades', 'Diamonds', 'Clubs', 'Hearts']:
        train_suits.append(Train_suits())
        train_suits[i].name = Suit
        filename = Suit + '.jpg'
        train_suits[i].img = cv2.imread(
            filepath+filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    return train_suits


def preprocess_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    img_h, img_w = np.shape(image)[:2]
    bkg_level = gray[int(img_h/100)][int(img_w/2)]
    thresh_level = bkg_level + BKG_THRESH

    retval, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)

    return thresh


def find_cards(thresh_image):
    dummy, cnts, hier = cv2.findContours(
        thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(
        range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]), reverse=True)

    # If there are no contours
    if len(cnts) == 0:
        return [], []

    # Otherwise, initialize empty sorted contour and hierarchy lists
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts), dtype=int)

    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i], True)
        approx = cv2.approxPolyDP(cnts_sort[i], 0.01*peri, True)

        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
                and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card


def preprocess_card(contour, image):

    # Initialize new Query_card object
    qCard = Query_card()

    qCard.contour = contour

    # Find perimeter of card and use it to approximate corner points
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01*peri, True)
    pts = np.float32(approx)
    qCard.corner_pts = pts

    # Find width and height of card's bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    qCard.width, qCard.height = w, h

    # Find center point of card by taking x and y average of the four corners.
    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]

    # Warp card into 200x300 flattened image using perspective transform
    qCard.warp = flattener(image, pts, w, h)

    # Grab corner of warped card image and do a 4x zoom
    Qcorner = qCard.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
    Qcorner_zoom = cv2.resize(Qcorner, (0, 0), fx=4, fy=4)

    # Sample known white pixel intensity to determine good threshold level
    white_level = Qcorner_zoom[15, int((CORNER_WIDTH*4)/2)]
    thresh_level = white_level - CARD_THRESH
    if (thresh_level <= 0):
        thresh_level = 1
    retval, query_thresh = cv2.threshold(
        Qcorner_zoom, thresh_level, 255, cv2. THRESH_BINARY_INV)

    # Split in to top and bottom half (top shows rank, bottom shows suit)
    Qrank = query_thresh[20:185, 0:128]
    Qsuit = query_thresh[186:336, 0:128]

    # Find rank contour and bounding rectangle, isolate and find largest contour
    dummy, Qrank_cnts, hier = cv2.findContours(
        Qrank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea, reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query rank
    # image to match dimensions of the train rank image
    if len(Qrank_cnts) != 0:
        x1, y1, w1, h1 = cv2.boundingRect(Qrank_cnts[0])
        Qrank_roi = Qrank[y1:y1+h1, x1:x1+w1]
        Qrank_sized = cv2.resize(Qrank_roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
        qCard.rank_img = Qrank_sized

    # Find suit contour and bounding rectangle, isolate and find largest contour
    dummy, Qsuit_cnts, hier = cv2.findContours(
        Qsuit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Qsuit_cnts = sorted(Qsuit_cnts, key=cv2.contourArea, reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query suit
    # image to match dimensions of the train suit image
    if len(Qsuit_cnts) != 0:
        x2, y2, w2, h2 = cv2.boundingRect(Qsuit_cnts[0])
        Qsuit_roi = Qsuit[y2:y2+h2, x2:x2+w2]
        Qsuit_sized = cv2.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        qCard.suit_img = Qsuit_sized

    return qCard


def match_card(qCard, train_ranks, train_suits):
    best_rank_match_diff = 10000
    best_suit_match_diff = 10000
    best_rank_match_name = "Unknown"
    best_suit_match_name = "Unknown"
    i = 0

    if (len(qCard.rank_img) != 0) and (len(qCard.suit_img) != 0):

        for Trank in train_ranks:

            diff_img = cv2.absdiff(qCard.rank_img, Trank.img)
            rank_diff = int(np.sum(diff_img)/255)

            if rank_diff < best_rank_match_diff:
                best_rank_diff_img = diff_img
                best_rank_match_diff = rank_diff
                best_rank_name = Trank.name

        for Tsuit in train_suits:

            diff_img = cv2.absdiff(qCard.suit_img, Tsuit.img)
            suit_diff = int(np.sum(diff_img)/255)

            if suit_diff < best_suit_match_diff:
                best_suit_diff_img = diff_img
                best_suit_match_diff = suit_diff
                best_suit_name = Tsuit.name

    if (best_rank_match_diff < RANK_DIFF_MAX):
        best_rank_match_name = best_rank_name

    if (best_suit_match_diff < SUIT_DIFF_MAX):
        best_suit_match_name = best_suit_name

    return best_rank_match_name, best_suit_match_name, best_rank_match_diff, best_suit_match_diff


def draw_results(image, qCard):

    x = qCard.center[0]
    y = qCard.center[1]
    cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

    rank_name = qCard.best_rank_match
    suit_name = qCard.best_suit_match

    cv2.putText(image, (rank_name+' of'), (x-60, y-10),
                font, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, (rank_name+' of'), (x-60, y-10),
                font, 1, (50, 200, 200), 2, cv2.LINE_AA)

    cv2.putText(image, suit_name, (x-60, y+25),
                font, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, suit_name, (x-60, y+25), font,
                1, (50, 200, 200), 2, cv2.LINE_AA)

    return image


def flattener(image, pts, w, h):
    temp_rect = np.zeros((4, 2), dtype="float32")

    s = np.sum(pts, axis=2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=-1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    if w <= 0.8*h:  # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h:  # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    if w > 0.8*h and w < 1.2*h:  # If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0]  # Top left
            temp_rect[1] = pts[0][0]  # Top right
            temp_rect[2] = pts[3][0]  # Bottom right
            temp_rect[3] = pts[2][0]  # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0]  # Top left
            temp_rect[1] = pts[3][0]  # Top right
            temp_rect[2] = pts[2][0]  # Bottom right
            temp_rect[3] = pts[1][0]  # Bottom left

    maxWidth = 200
    maxHeight = 300

    dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1,
                   maxHeight-1], [0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect, dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

    return warp
