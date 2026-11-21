"""The asks: one entry per AI round, how that round is asked (docs/pipeline.html, "The method": every ask's settings
come from here, measured per round in the lab). An entry is keyed by the name the round's asks are recorded under
(watch.recorded, the lab's ROUNDS); watch.recorded passes that name to the ask as ``route``, and ai.ask asks by its
entry. A round that puts several things in one ask reads its own entry's ``items``.

An entry (Ask):
    model       the OpenRouter model
    provider    the one provider it is pinned to, with no fallbacks (ai.py: a fallback once answered without the
                pictures)
    efforts     the reasoning effort it is asked at, then the one it is asked once more at when the first attempt's
                reply does not come whole (ai.ask: its deadline passes, or the provider's stream ends before the model
                finished): the next lower, or low again for a round asked at low. A second such failure fails. One
                effort listed: no second attempt at a lower effort
    deadline_s  each send's deadline, by the wall clock from its turn to send: sending the pictures, the wait, reading
                the reply; waiting for a turn is not on it
    items       how many things one ask carries, for a round that batches them: round 2's pieces to a sheet, twin
                sets, bones, the build pick's part pictures; None for a round that asks about one thing
    request_bytes the most bytes one request may carry, for a round that shrinks and splits its pictures to fit
                (the eyes); None for a round that doesn't
    picture     max_edge, format and quality of the sent copy; record pictures are left alone. A tuple of edges
                follows attachment order, repeating its last edge (whole views/sheets 1280, close-ups 1024).
    measured    the numbers that chose it, from the lab
    date        when they were measured

Every ask names its round: a route with no entry is an error, not the default. Numbers that only lay out a picture
stay with the drawing that uses them, since a round's ask carries every sheet it draws: how many tiles go on one sheet
(the eyes' TILES_PER_SHEET 12 and REGIONS_PER_SHEET 20, the roster's PER_SHEET 6, round 2's COLUMNS), tile sizes.

Picture encoding and size are checked by eye and with the pilot replay; record drawings stay unchanged. Other new
settings for a round are measured in the lab first (rigswap lab score, on the round's truth subset, two repeats or
more, against the entry's setting asked at the same time) and change here only where they are within noise of the
entry's accuracy and stability: within the spread between two runs of the entry's own setting, as the lab has
measured it for that round (round 2: about 4 of 165 answers; round 3: 11 of 306; round 4: 1.1 points of the model's
height), and no worse on wrong answers, failed asks or the same answer across repeats.

Measured 2026-10-12, each round's lab subset, 2 repeats, the pictures and questions the pipeline asks today, against
GLM 5.3 Flash from Together at high (lab notes kept outside this repo): medium and
low, Qwen 3.7 Flash (Alibaba) and Gemini 2.5 Flash-Lite (Google AI Studio) at high. The cheaper models lost on every
round with a truth but the held item's handle: round 1 195 and 207 of 240 against 236, round 2 510 and 472 of 620
against 558 and 568, round 3 144 and 135 of 204 against 175, round 4 11% and 21% of the height from the truth against
6-8%; and Gemini at high thinks so long it costs more than GLM. A GLM ask's cost is mostly its pictures (round 2: 11,400
tokens in, 500 to 1,600 out), so lower effort saves little money; it saves time.
"""
from __future__ import annotations

from dataclasses import dataclass

EFFORTS = ('high', 'medium', 'low')       # from most thinking to least


@dataclass(frozen=True)
class Picture:
    max_edge: int | tuple[int, ...] = 1280
    format: str = 'JPEG'
    quality: int = 85


@dataclass(frozen=True)
class Ask:
    model: str = 'z-ai/glm-5.3-flash'
    provider: str = 'together'
    efforts: tuple = ('medium', 'low')     # 2026-10-12: medium by default (the owner's call: speed and cost over the
                                            # lab's margin); round 1 stays at high (one cheap ask that sets up and front).
                                            # Low was run on the three heavy models (opt15): spiderman's torso piece
                                            # was answered a rendering layer and dropped, so he got Mario's hands;
                                            # goku was fine and a third faster; tikal unchanged. Medium stays.
    deadline_s: float = 300
    items: int | None = None
    request_bytes: int | None = None
    picture: Picture = Picture()
    measured: str = ''
    date: str = ''

    def effort(self):
        """The effort the first attempt is asked at."""
        return self.efforts[0]

    def lower(self, effort):
        """The effort an attempt at ``effort`` is asked once more at when its deadline passes: the next of
        ``efforts`` after it, None after the last."""
        at = self.efforts.index(effort) if effort in self.efforts else None
        if at is None:          # an effort the round does not list (an ask that set its own): the next one down
            at_all = EFFORTS.index(effort)
            return EFFORTS[at_all + 1] if at_all + 1 < len(EFFORTS) else None
        return self.efforts[at + 1] if at + 1 < len(self.efforts) else None


# What every round was asked with before the rounds were measured apart (2026-10-10): GLM 5.3 Flash from Together,
# effort high. Left unset the model's thinking has no bound (on the round-2 question for nine pieces it reasoned for
# 37,911 tokens and 349 s, past the deadline); 'medium' read round 1's up wrong for two downloads of three (kritter -y,
# majorfirepower +y) where 'high' read it right; 'high' with four pieces to a round-2 sheet answered all 48 sheets of
# three downloads within 93 s. Rounds with no truth to measure against keep it: the eyes, the build's pick of a
# variant, choosing a target.
DEFAULT = Ask(measured='round 1 at medium read up wrong for 2 of 3 downloads; high answered all 48 round-2 sheets of '
                       '3 downloads within 93 s', date='2026-10-10')

ASKS = {
    # The new middle (rigswap.core): the front's sign, a choice between the two fronts the rig's mirror axis allows,
    # one picture of each side. Not lab-measured yet.
    'core front': Ask(efforts=('medium', 'low'), picture=Picture(1280), date='2026-10-25'),
    # The look that fills the body of the record (core/fill.py): the character from four sides, two sides with every
    # joint numbered, a table of the joints; one answer names the body's joints and its front. Lab 2026-11-09
    # (five models, three tries each, against records written by hand): Gemini 3.7 Flash
    # named every key joint right or one joint off, 15 of 15, at about 1 cent a fill; GPT-6 Luna 13 of 15 at 0.19c;
    # Haiku 5.5 12 of 15; GLM 5.3 Flash 10 of 15, a third of its answers changing between tries. No provider pinned:
    # Google alone serves it.
    'core fill': Ask(model='google/gemini-3.7-flash', provider=None, efforts=('medium', 'low'), deadline_s=180,
                     picture=Picture(1280), measured='15 of 15 key joints right or one off, ~1c a fill',
                     date='2026-11-09'),
    # The same look asked once more of another model, when the first answer could not be used or the tripwire called
    # the draft a wreck: a second model's mistakes are its own (the lab: Luna and Gemini missed different models).
    'core fill again': Ask(model='openai/gpt-6-luna', provider=None, efforts=('medium', 'low'), deadline_s=180,
                           picture=Picture(1280), measured='13 of 15 key joints right or one off, 0.19c a fill',
                           date='2026-11-09'),
    # One limb of a skeleton fitted inside a file that had none, asked again on a picture of that limb alone, drawn
    # square to it (core/fill.py closer()): the look's first answer names a limb's joints a dot or two off on two
    # whole-body pictures, and a limb that points at the camera hides its bend there (Goku on Mario, size-2: the
    # right elbow named one dot from the hand, the forearm stretched five times over). Not lab-measured: set as the
    # fill, and its second model as the fill's.
    'core fill limb': Ask(model='google/gemini-3.7-flash', provider=None, efforts=('medium', 'low'), deadline_s=120,
                          picture=Picture(1280), date='2026-11-10'),
    'core fill limb again': Ask(model='openai/gpt-6-luna', provider=None, efforts=('medium', 'low'), deadline_s=120,
                                picture=Picture(1280), date='2026-11-10'),
    # The tripwire (core/tripwire.py): the draft in two of the fighter's frames beside the source; is it upright, is
    # every part attached. Lab 2026-11-09 (lab2): every model called the wrecked draft broken; Luna, Gemini and Haiku
    # raised that alarm on no corrected port (GLM on 2 of 13). No model's "good" can be trusted, so it is not asked.
    'core tripwire': Ask(model='openai/gpt-6-luna', provider=None, efforts=('medium', 'low'), deadline_s=120,
                         picture=Picture(1280), measured='wreck called 2 of 2, no alarm on 13 corrected ports',
                         date='2026-11-09'),
    # A complaint (core/complaint.py): the source, the draft and the user's words; which one entry of the record to
    # change. Lab 2026-11-09 (lab2): Luna named the fault written down beforehand on 13 of 16 known-bad drafts, the
    # others 3 to 6. Not lab-measured as a complaint.
    'core complaint': Ask(model='openai/gpt-6-luna', provider=None, efforts=('medium', 'low'), deadline_s=180,
                          picture=Picture(1280), measured='named the main fault on 13 of 16 known-bad drafts',
                          date='2026-11-09'),
    # Which of the model's pieces a complaint means ("take the jetpack off", "his hat is missing"): each piece the
    # costume draws, or each it leaves out, alone in its colours on a numbered tile, with the person's words and the
    # pieces' names (core/pieces.py). Not lab-measured; set as core complaint.
    'core complaint piece': Ask(model='openai/gpt-6-luna', provider=None, efforts=('medium', 'low'), deadline_s=180,
                                picture=Picture(1280), date='2026-11-10'),
    # Which fighter a model suits when no one names one (plugins/melee/pick.py): the model from three sides, the
    # fighters there are targets for. Not lab-measured; set as core front.
    'melee fighter': Ask(efforts=('medium', 'low'), picture=Picture(1280), date='2026-10-29'),
    # Which way is up: one side view turned four ways (rips of some games come in lying down).
    'core up': Ask(efforts=('medium', 'low'), picture=Picture(1280), date='2026-10-26'),
    # Which of the model's pieces the everyday look leaves out (face expressions, hand poses, a prop in the hand and on
    # the back, coarser copies): the whole model from four sides and every piece on a numbered tile, one answer
    # (core/look.py). Lab 2026-11-19 (strategy-2026-11-18/lab: run look-1's 10 asks, twice each, against keep lists
    # written first): answers that left out a body or outfit part, of 20, Gemini 3.7 Flash 2 (Inkling's eye mask, one
    # Vulpix piece; no other form kept), Hydrogen 2 at twice the price, GLM 6 (+10 forms kept), Luna 7 (+22), Luna at
    # high 9; Opus 0 and Sol 1 on the subscriptions. ~1c an ask, median 18 s.
    'core look': Ask(model='google/gemini-3.7-flash', provider=None, efforts=('medium', 'low'), deadline_s=180,
                     picture=Picture(1280), measured='left out a body or outfit part in 2 of 20 answers',
                     date='2026-11-19'),
    # Which of the parts other files place on the body (hairs, bottoms, tanks: grouped by the joints they ride) the
    # everyday look wears: a picture per group, each part alone on the grey body, three asks (core/assemble.py). Not
    # lab-measured; set as the old 'core variants' was (the default model).
    'core parts': Ask(efforts=('medium', 'low'), picture=Picture(1280), date='2026-10-26'),
    # Which joint the everyday look hangs each object of a file of its own on (her tank's ink, a broom), or none, and
    # the colour of its surfaces drawing no image: a picture per object, a tile per joint carrying nothing, the object
    # seen through what covers it, three asks (core/assemble.py attach). Not lab-measured. High, as core paint: in
    # words at medium, two answers of three hung the Inkling's ink on Tank_Bomb (at-2); at high, in words, a coin.
    'core attach': Ask(efforts=('high', 'medium'), picture=Picture(1280), date='2026-10-28'),
    # What each twin pair is to the body (legs, a four-legged body's front legs, arms, or other): a picture per six
    # pairs, the grey body from the front, side and back with each pair in its colour and numbered, three asks
    # (core/labels.py). Not lab-measured; set as the hang ask before it was (core parts).
    'core labels': Ask(efforts=('medium', 'low'), picture=Picture(1280), date='2026-10-27'),
    # What each chain off the body is (hair, cloth, a tail, armour, a prop, part of the face...) and whether the bind
    # pose holds hair or cloth out: the pairs' pictures, each chain alone close in its own colours, three asks
    # (core/labels.py chains; it replaced the hang ask). Not lab-measured; set as core labels.
    'core chains': Ask(efforts=('medium', 'low'), picture=Picture(1280), date='2026-10-27'),
    # Which of the fighter's slot kinds (a sword, a shield) each prop the model holds or carries is, or none: the props'
    # pictures as the chains', three asks (core/labels.py equipment), only where the fighter's hands hold more than one
    # kind. Not lab-measured; set as core chains.
    'core equipment': Ask(efforts=('medium', 'low'), picture=Picture(1280), date='2026-10-28'),
    # Whether two pieces of a segmented rip that meet are joined or only touching (a hand resting on a thigh): a
    # picture per contact on a loop through the trunk, its two pieces in two colours, three asks (core/segments.py
    # touching). Not lab-measured; set as core labels.
    'core touching': Ask(efforts=('medium', 'low'), picture=Picture(1280), date='2026-10-28'),
    # Which way each limb labels named goes where its bones branch (on along a branch toward its hand or foot, or its
    # hand or foot is there): a row per fork seen from two sides, its branches in colours, three asks (core/forks.py).
    # Not lab-measured; set as core labels.
    'core forks': Ask(efforts=('medium', 'low'), picture=Picture(1280), date='2026-10-27'),
    # Which of the head's surfaces (a mesh with one material) are the eyes, and which pictures found in the textures
    # show them half closed and closed: the head numbered, each surface alone, the pictures found (core/eyes.py).
    # Not lab-measured. High, as the old eyes round was asked: one ask a swap; at medium two asks on boomerangbro's
    # pictures (the head drawn twice, its labels moved) named its half and closed eyes once and left them out once.
    'core eyes': Ask(efforts=('high', 'medium'), picture=Picture(1280), date='2026-10-25'),
    # Which of a download's unloaded images is the everyday iris of eyes whose file names only their white (a rip
    # that carries the second UV map as a copy of the eye mesh, its files not named as Ultimate's are): each image
    # laid over the white, one ask a swap (core/eyes.py, ask_iris). Not lab-measured; set as core eyes.
    'core eyes iris': Ask(efforts=('high', 'medium'), picture=Picture(1280), date='2026-11-10'),
    # Which image beside a colour image, if any, is a mask where the game paints a colour of its own in play (a team's
    # ink), and in what colour: the model in its colours, then each colour image with the images sharing its name,
    # three asks (core/paint.py). Not lab-measured. High, as the eyes are: at medium two answers of three took Bowser
    # Odyssey's metalness map and Psyduck's shading layers for masks (pt-1, pt-p2).
    'core paint': Ask(efforts=('high', 'medium'), picture=Picture(1280), date='2026-10-27'),
    # Whether a mask paints its dark where the files contradict themselves (the colour image blank only under it): the
    # model in its colours, then the parts drawing each such image as drawn and painted, close, three asks, all must
    # paint (core/paint.py, dark_ways). Not lab-measured; set as core paint.
    'core paint way': Ask(efforts=('high', 'medium'), picture=Picture(1280), date='2026-10-27'),
    # The colour a layer shows in, where its image (in greys, no alpha) says only where it shows: the model in its
    # colours, then each layer's image, three asks (core/paint.py, show). Not lab-measured; set as core paint.
    'core layer': Ask(efforts=('high', 'medium'), picture=Picture(1280), date='2026-10-27'),
    # Whether an image's alpha cuts what draws it, where its material wires no alpha: a row per image, the parts drawing
    # it close from the front and the back, cut and whole, three asks (core/opacity.py). Not lab-measured; set as core
    # paint way, whose rows and answers it shares.
    'core opacity': Ask(efforts=('high', 'medium'), picture=Picture(1280), date='2026-10-28'),
    # 16 downloads x 2: high 236 of 240 fields right, 118 of 120 the same in both repeats; medium 227 and 114, low 226
    # and 110 (up, front and the others files), so high stays.
    # Wording lab (2026-10-13). All 49 truth cases,
    # two uncached repeats per wording, interleaved with the current question at identical settings. Judge limbs
    # rather than trunk shape, including short limbs, straight poses and joints hidden under clothing. Same total
    # correct and wrong; stability two fields lower, within the 357-359 range of the three paired controls.
    # Body type alone 97/98 versus 98/98: one Kevin-Zelda answer still ball; Kevin-Mario biped twice. Earlier wordings
    # in astra-bodytype-lab and astra-bodytype-lab-can-bend got body type 96/98 and 94/98 and were not shipped.
    'describe round 1': Ask(efforts=('high', 'medium'),
                            measured='body wording lab, 49 cases x 2: current/new 723/736 right each (13 wrong each); '
                                     'repeats 361,362 / 365,358; stable 359/368 / 357/368 '
                                     '(paired controls 357-359); body type 98/98 / 97/98; failed asks 0/0. '
                                     'Effort lab, 16 downloads x 2: high 236/240 right, 118/120 stable; '
                                     'medium 227,114; low 226,110', date='2026-10-13'),
    # 16 sheets x 2, two runs of each: high 558 and 568 of 620 right (277 and 288 of 310 stable), low 552 and 536 (276
    # and 272), medium 534 (262): low is 3 points under high on the mean of two runs each, more than either's own
    # spread (10 and 16 answers), so high stays; low answered in 5-6 s against 10-12 (p90 8-10 against 26-36).
    # Four pieces to a sheet, one ask each (describe/pieces/pictures.py PAGE), as on 2026-10-10: nine pieces to an ask
    # reasoned for 37,911 tokens and 349 s, past the deadline; four answered all 48 sheets of three downloads within
    # 93 s. Not measured since against more.
    # Tikal's pilot sanity check changed 14/46 moves_as at 1280/q85: retain more detail in the numbered sheets.
    # 2026-10-13: high again (the medium default of 2026-10-12 cost the Minion his hands: at medium round 2 called nine
    # of his pieces hanging, the hands among them, where high called six; the lab's "three points" hid that kind).
    'describe round 2': Ask(efforts=('high', 'medium'), picture=Picture(1024, quality=85), items=8, measured='pieces lab, 16 sheets x 2, two runs: high 558, 568/620 right, 277, '
                                              '288/310 stable; low 552, 536 and 276, 272; medium 534, 262; 4 pieces '
                                              'to a sheet as on 2026-10-10', date='2026-10-12'),
    # Up to 6 twin sets to a picture, one ask each, against one sheet's sets (1 or 2): on the twins lab's 28 cases, 43
    # sets with truth from two answerers, 2 repeats, 67 right, 0 wrong, 19 can't tell, the same in both repeats on 36
    # sets, against 66, 0, 20 and 37 asked a sheet at a time (3 to a picture: 63, 0, 23, 40); every unstable set is
    # golduck's or tikal's, whose copies look the same (either is right, and can't tell keeps one that draws best).
    # goku-jump asks 6 times instead of 20, golduck 8 instead of 24. Medium and low named a wrong copy (1 and 2 of 86)
    # where high named none, so high stays.
    'describe twins': Ask(items=6, measured='twins lab, 43 sets x 2: 6 to an ask 67 right / 0 wrong / 19 cant, 36 '
                                            'stable; a sheet to an ask 66 / 0 / 20, 37 stable; medium 1 wrong, low 2',
                          date='2026-10-12'),
    # The held item's handle, 9 cases x 2: 18 of 18 right and every case the same in both repeats at high, medium and
    # low alike (Qwen too; Gemini 14); low answered in 2 s, p90 2 s against 7.
    'describe held': Ask(picture=Picture((1280, 1280, 1024)), efforts=('low', 'low'), measured='held-ask lab, 9 cases x 2: high, medium and low each '
                                                          '18/18 right, 9/9 stable; low $0.00033 an ask against '
                                                          '$0.00055', date='2026-10-12'),
    # Up to 20 bones to an ask, against 10: on the bones lab's 14 asks (102 bones with truth), drawn again with the
    # current code from the same downloads, 182 of 204 answers right, 96 of 102 bones the same in both repeats, 185
    # the same downstream, no ask failed, against 175, 87, 185 and 2 of 28 failed (a deadline, a dropped connection) at
    # 10; the 29 truth bones that moved into a bigger ask (17 to 19 bones, all named ones) got 54 of 58 either way.
    # Asks per run 140 to 109 over 11 downloads (spiderman 69 to 46), cost per bone 19% lower. A close-up of more than
    # ten bones repeats its ten tints. Medium (157) and low (166) lost to high, so high stays.
    # q92 keeps the tiny numbered circles and bone tints legible; resized PNGs grew from antialiased edges.
    # The pilot sanity check changed more roles than the lab's noise, so retain more pixels in this route only.
    # 2026-10-12: 600 s. At high the model reasons for up to 26k tokens on a hard group (tikal's trunk and legs) and
    # passes 300 s with little queue time (4 of 7 asks in one run, 2 of 7 in another); the medium answer it then fell
    # to read tikal's whole left leg as "other", and the build collapsed. The time is the model's, so it gets it.
    # Pictures stay at 1500/1280 q92 here. Measured on tikal's six asks at high, same question (2026-10-12 evening):
    # at these sizes 2 of 6 reasoned past Together's 300 s cut and fell to medium, at 1024/q85 4 of 6; where both
    # answered at high the larger pictures agreed with pilot 8 as well or better (8/9 against 4/9, 7/8 against 6/8).
    # Whole views at 1024, close-ups at 1280, all q92 (lab 2026-10-12 late, tikal's six asks at high, same question): against
    # whole views at 1500, agreement with pilot 8 within noise (7/14 9/9 7/9 5/8 2/7 against 9/14 8/9 9/9 4/8 3/7), one
    # of six cut at 300 s either way, prompt tokens a quarter fewer; the saving lands on many-ask rigs (spiderman).
    'describe round 3': Ask(efforts=('high', 'medium'), deadline_s=600, picture=Picture((1024, 1024, 1024, 1280), quality=92), items=20, measured='bones lab, 102 bones x 2: 20 to an ask 182/204 right, 96/102 stable, '
                                               '0 failed; 10 to an ask 175, 87, 2/28 failed; medium 157, low 166',
                            date='2026-10-12'),
    # 13 asks x 2, two runs of each: high placed the joints 7.6% and 6.2% of the height from the truth's (median; 8.5%
    # and 7.3% mean; the 90th percentile between repeats 11.6% and 11.3%), low 7.8% and 7.7% (9.3%, 8.7%; 13.3%,
    # 15.6%), medium 7.8% and 8.9% (9.2%, 10.8%; 15.6%, 16.8%): low is a point farther and less steady in both runs,
    # at the edge of the round's noise, so high stays. Low answers in 8 s (p90 11-12) against 18-19 (59-75).
    # Grid labels were legible by eye at q85; the frozen-cell check moved joints beyond the lab's height noise,
    # so keep more detail, at q92, for this route as well.
    # 2026-10-13: high and 1500/1280 q92 again (the medium default and the 1024/q85 cut of 2026-10-12 were never
    # measured for this round): the two boneless models of the 32 (the Minion, Budokai Goku) came apart in game with
    # round 4 answering in 2 to 9 s on a few hundred reasoning tokens, where pilot 8 took 17 to 131 s at high.
    # Elbows/knees on the existing limb close-ups; every joint in the top view, with per-view hidden answers.
    # Paired current-entry comparison (the elbow lab). Same frozen current-format
    # pictures, 13 truth asks (two answerers), two uncached repeats each. The one front-only answer is counted as
    # wrong below, not silently excluded; the established distance metric does not run one_view's mesh depth.
    # Assigned-piece candidates: astra-joints-derive-lab, 15 asks x two uncached repeats per change against
    # b0939a5, identical settings. Marks alone: median/mean/repeat p90 6.04/7.33/9.39% height. A separate
    # confirmation paragraph failed stability (11.41%) and one schema; the direct confirm-or-cell instruction
    # below passes against the frozen baseline, though its repeat p90 is 1.88 points above marks alone.
    # Truth: the 13 existing cases plus both Goku arms, with inspected mesh shoulder/elbow/wrist references.
    'describe round 4': Ask(efforts=('high', 'medium'), picture=Picture((1500, 1500, 1500, 1280), quality=92),
                            measured='candidate joints lab, 15 asks x 2: baseline/confirm-or-cell median distance '
                                     '6.20%/5.50% of height, mean 7.27%/7.22%, repeat p90 10.21%/11.27%; '
                                     'each within the established 1.1-point band against current code. '
                                     '116/116 triangulated truth answers, >10% away or unplaced 27/28, '
                                     'failed asks 0/0, invalid cells 2/0. Overall improvement unproven. '
                                     'Two no-answer transport failures recovered with identical requests; '
                                     'all attempts retained. Settings unchanged.', date='2026-10-13'),
    # No truth. A request of at most 900,000 bytes: the eyes shrink and then split their pictures to fit
    # (details/eyes/questions.py), since OpenRouter rejected a 14-picture eyes request at about 1.5 MB (2026-10-10). Other rounds'
    # asks are not held to it: round 4's grid asks are 1-4 MB and go through.
    'describe eyes': Ask(picture=Picture((1024, 1280)), request_bytes=900_000, measured=DEFAULT.measured + '; a 14-picture eyes request of ~1.5 MB '
                                                         'rejected by OpenRouter', date=DEFAULT.date),
    # No truth: 8 cases x 2, the same in both repeats on 6 at high, medium and low; low agreed with today's answers on
    # 12 of 16, medium on 8, so high stays.
    'build place held': Ask(picture=Picture(1024), measured='held lab, 8 cases x 2, no truth: 6/8 stable at high, medium, low; agreeing '
                                     "with today's answers 14/16, 8/16, 12/16", date='2026-10-12'),
    # No truth. At most 12 part pictures to one ask, the parts split over asks of equal size past that
    # (build/pick.py), as the pick has asked since it was written; not measured.
    'build pick': Ask(picture=Picture(1024), items=12, measured=DEFAULT.measured + "; 12 pictures to an ask: today's, not measured",
                      date=DEFAULT.date),
    # The whole picture and at most 11 close-ups of where the variants differ, 12 pictures in its one ask; the rest
    # listed as not pictured (not measured). Its criterion (2026-10-13): which build looks most like the character as
    # it came, at rest and in every test pose, and bends like a real body; "its limbs and joints where the character's
    # are" dropped, since it decided against a build whose target joints were not where the character's are on
    # purpose, before the pictures were looked at. Lab:
    # pick-variant-r34, 22 pilot 5/6 cases (joint variants), truth from two answerers who agree on 15 (4 contested, 3
    # "can't tell"), two uncached repeats of each wording at this entry's settings, side by side: of 30 answers,
    # right 10 and 10, wrong 16 and 16, can't tell 4 and 4; as used (can't tell takes the default) 11 and 12 right;
    # the same letter in both repeats 9 and 12 of 22. Within noise or better: shipped.
    'build pick a variant': Ask(picture=Picture((1280, 1024)), items=12,
                                measured='variant-pick lab, 22 cases (15 with agreed truth) x 2, current/new criterion: '
                                         "right 10/10, wrong 16/16, can't tell 4/4 of 30; as used 11/12 right; same "
                                         'letter both repeats 9/22 vs 12/22; picture frame lab 2026-10-13, '
                                         'current/extended view, '
                                         "22 cases x 2 at identical settings: right 13/9, wrong 12/13, can't tell "
                                         '5/8 of 30, as used 15/13, stable 10/13 of 22, failures 0/0. '
                                         '15 cases have byte-identical requests: right 10/7, as used 11/9; '
                                         'on the 7 changed cases, right 3/2, wrong 3/3, as used 4/4 of 10, '
                                         'stable 4/6 of 7: within the identical-picture noise. Orthographic '
                                         'and camera-dolly trials rejected (right 7 vs 11 of 30 each). '
                                         'Evidence: astra-lab/pick-frame-overscan. Source at target rest height, '
                                         'picture-only lab 2026-10-13: 2 uncached repeats each, same settings, '
                                         "original 22 cases current/new: right 12/12, wrong 14/13, can't tell 4/5 "
                                         'of 30; as used 14/14; stable 6/9 of 22, failures 0/0. Five cases have '
                                         'small rest-height differences (0.027-2.314% of target height), none '
                                         'like Kevin/Zelda, added with measured silhouette truth C or F: '
                                         'source/C/F height-width 1.16555, other builds 1.65458-1.65465. '
                                         'Including Kevin: right 14/12, wrong 14/15, cant tell 4/5 of 32; '
                                         'as used 16/14, stable 6/9 of 23, failures 0/0. Baseline repeats '
                                         'as used 9/7 and wrong 6/8: pooled loss 2 right-as-used and 1 extra '
                                         'wrong within observed repeat noise; no accuracy gain demonstrated. '
                                         'Kevin current F/C, new A/E: both new answers wrong despite correct '
                                         'reference scale. Evidence: astra-lab/came-height/metrics.json and '
                                         'height-audit.json; tie schema lab 2026-10-13, 23 cases including Kevin, '
                                         '16 with agreed/acceptable truth, two uncached repeats per side at identical '
                                         'settings and pictures: current/new right 12/12, wrong 14/10, cant tell '
                                         '6/10 of 32; as used 15/20 right (repeats 7,8 / 10,10); stable letters '
                                         '16/16 of 23, stable selections 17/18, failures 0/0. Schema names the tie '
                                         'in between; question text unchanged. 21 named ties in 46 new answers. '
                                         'Kevin right 1/2 each: current F/E, new D/F. Pilot 5 bending recomputed '
                                         'from frozen pose measures for both defaults (old records only had tearing). '
                                         'An added prose instruction was rejected: right 14/7, as used 16/15. '
                                         'Evidence: astra-lab/tie-rank-schema/metrics.json (shipped) and '
                                         'astra-lab/tie-rank/metrics.json (rejected). Source in every row and close-up, '
                                         'at the base fits ankle-to-head-top scale, picture lab 2026-10-14: 23 cases '
                                         'including Kevin, two uncached repeats per side at identical settings, schema '
                                         'and question (layout legend corrected for both). Current/new right 10/12, '
                                         'wrong 10/13, abstained or defaulted 12/7 of 32; as used 19/18; stable letters '
                                         '12/11 and selections 15/12 of 23; failures 3/0. Per-repeat as used 9,10 / '
                                         '11,7 and wrong 6,4 / 5,8: mean loss 0.5 used-right and 1.5 extra wrong '
                                         'within baseline repeat spreads 1 and 2; no overall accuracy gain demonstrated. '
                                         'Kevin current A/F, new F/F (C/F accepted). Close-up source cameras centre on '
                                         'the sources own rest spans, at unchanged magnification. Rejected shared-crop '
                                         'trial: as used 21/17, wrong 7/14, stable 14/10; centred trial with stale legend: '
                                         'as used 13/13, wrong 14/18, stable 11/16. Evidence: '
                                         'astra-lab/came-everywhere-final/metrics.json, came-everywhere/metrics.json '
                                         'and came-everywhere-focused/metrics.json; 12 pictures to its ask: not measured',
                                date='2026-10-14'),
    'choose a target': DEFAULT,
    'melee physics': Ask(efforts=('medium',), deadline_s=300, items=None,
                         measured='One picture judgement only when the game has fewer free chains than the model',
                         date='2026-10-12'),
}


def entry(route):
    """The Ask a route names: an Ask is itself, a name is its entry in ASKS. Every ask names its round: none, or a
    name with no entry, is a ValueError."""
    if isinstance(route, Ask):
        return route
    if route not in ASKS:
        raise ValueError(f'no entry in library/asks.py for the ask {route!r}: every ask names its round')
    return ASKS[route]
