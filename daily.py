import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
from numpy import polyfit, polyval
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import math
from scipy import stats
from scipy.signal import find_peaks

# extract data
# list of daily tolls up to today
# list of added toll for each day after
# work out proportions of each day after vis a vis total
# use that to work out true totals
# graph true totals, work out line of best fit, assume peak is mean of normal dist, work out sd
# create normal distribution for true totals, use z scores to make histogram
# predict true totals for each day
# using fraction list, predict daily announcement

data = pd.read_csv(r'C:\Users\james\Documents\Daily.csv',  na_filter= False)
eng_total = int(data.columns[-1])
ons_multiplier  = 0.3
full_total = 34796
eng_prop = float(1/(eng_total/full_total))
eng_prop = eng_prop * (1+ons_multiplier)
adict_of_lists = {}
dict_of_lists = {}
# dictionary of deaths per day
n=0
mmer = 0
cn = datetime.date(2020, 3, 1)
for column_name in data.columns:
    if mmer < 86:
        if n > 0:
            start = n+5
            temp_list = data[column_name].tolist()
            st_format_list = temp_list[start:]
            final_list = []
            for i in st_format_list:
                if i:
                    final_list.append(i)
                if not i:
                    break
            if final_list:
                adict_of_lists[cn] = final_list
            cn += datetime.timedelta(days=1)
        n+=1
    mmer += 1

for each in adict_of_lists:
    hold_list = []
    for every in adict_of_lists[each]:
        every = float(every)
        hold_list.append(every * eng_prop)
    dict_of_lists[each] = hold_list

dict_sum_by_date = {}
list_dates2 = [*dict_of_lists]
list_dates2.sort()
for d2 in list_dates2:
    dict_sum_by_date[d2] = sum(dict_of_lists[d2])

dict_of_totals = {}
# dict of deaths recorded after by  each day after

temp_list=[]

for k in dict_of_lists:
    day_after = 1
    for l in dict_of_lists[k]:
        dict_of_totals[day_after] = []
        day_after+=1
for k in dict_of_lists:
    day_after = 1
    for l in dict_of_lists[k]:
        temp_list = dict_of_totals[day_after]
        temp_list.append(l)
        dict_of_totals[day_after] = temp_list
        day_after += 1

sum_days_after = []
anchor = 1
for day in dict_of_totals:
    if day == anchor:
        results = dict_of_totals[day]
        results = [float(i) for i in results]
        b = sum(results)
        sum_days_after.append(b)
        anchor += 1
#      this is ratio of each day to total, but misleading as skewed towards days with more data


true_list = []
# for each date, gives final reading by adding on expected proportion for each future day's recordings

daily_sum_list = []
to_sum = []
fraction_list = []

# make list of recorded total toll for each day, to be inflated for true total. need to sum up values
# in first dict,

dn = datetime.date(2020, 3, 1)
for k in dict_of_lists:
    if k == dn:
        to_sum = dict_of_lists[k]
        fl = [float(i) for i in to_sum]
        res = sum(fl)
        daily_sum_list.append(res)
        dn += datetime.timedelta(days=1)

# work out max days after

no_days_after = len(sum_days_after)-1

# y is recorded sum for each day
# add on extra days, which will increase by one each loop

first_real = datetime.date(2020, 4, 1)
ch = first_real
ch -= datetime.timedelta(days=14)
real_dict = {}

inter = ch

for g in dict_of_lists:
    if g == inter:
        real_dict[g] = dict_of_lists[g]
        inter += datetime.timedelta(days=1)


tdy = datetime.date.today()
rangefinder = len(real_dict) -14
very_end_date_1 = (max(real_dict))

# summ up all the days with complete data ; ie last announced -14 days as 15th will be full
sum_of_real_trues= []
day_after_real = {}

full_t_dict = {}

sigh = 0
inter_2 = ch
for f in real_dict:
    if f == inter_2:
        full_t_dict[f] = real_dict[f]
        inter_2 += datetime.timedelta(days=1)
        sigh+=1
        if sigh == rangefinder:
            break

middle_dict = {}
mdl = [*full_t_dict]
mdl.sort()
for mdll in mdl:
    middle_dict[mdll] = sum(full_t_dict[mdll])

real_sum = 0

leng = len(full_t_dict)
dict_real_separator = {}

randy = 1
another = ch

for ran in range(leng):
    if randy > 14:
        dict_real_separator[another] = 15
    else:
        dict_real_separator[another] = randy
        randy += 1
    another += datetime.timedelta(days=1)

truly_real_fractious_dict = {}

come_down = 15
for humpy in range (come_down):
    truly_real_fractious_dict[humpy+1] = []

for w in full_t_dict:
    sumbo = sum(full_t_dict[w])
    real_sum += sumbo
    no_to_do = dict_real_separator[w]
    for nos in range (no_to_do):
        neos = 16-(nos+2)
        truly_real_fractious_dict[neos+1].append(full_t_dict[w][neos])
    come_down -= 1

true_real_fract_dict_sum = {}
for fr in truly_real_fractious_dict:
    t_list = sum(truly_real_fractious_dict[fr])
    true_real_fract_dict_sum[fr] = t_list

dict_real_sum_by_date = {}
list_dates1 = [*full_t_dict]
list_dates1.sort()
llist_of_sums_by_date = []
list_of_sums_by_date = []
for d1 in list_dates1:
    llist_of_sums_by_date.append(dict_sum_by_date[d1])

if len(llist_of_sums_by_date) > 15:
    excess = len(llist_of_sums_by_date) - 15
    excess_sum_list = llist_of_sums_by_date[14:(14+excess)]
    excess_sum_list = sum(excess_sum_list)
    list_of_sums_by_date = llist_of_sums_by_date[0:15]
    list_of_sums_by_date[14] = excess_sum_list
else:
    list_of_sums_by_date = llist_of_sums_by_date

dict_sum_days_whereXisreal = {}
lndays = len(list_of_sums_by_date)-1
rev_list_of_sums_by_date = list_of_sums_by_date[::-1]
num=15
for sno in range(lndays, -1, -1):
    holder = []
    for snot in range(sno, -1, -1):
        holder.append(rev_list_of_sums_by_date[snot])
    dict_sum_days_whereXisreal[num]=holder
    num-=1
for yikes in dict_sum_days_whereXisreal:
    dict_sum_days_whereXisreal[yikes] = sum(dict_sum_days_whereXisreal[yikes])

if len(dict_sum_days_whereXisreal)<15:
    ff = 15-len(dict_sum_days_whereXisreal)
    for ffs in range(1, ff+1):
        dict_sum_days_whereXisreal[ff] = 0

# real sum is sum of all days with at least one real figure and 15 days of totals
# true_real_fract.._sum is a dict of sums of real figures for days after
# dict_whereX is dict of totals of every day where X is a real number


not_complete_days_dict = {}
beginner = very_end_date_1
beginner -= datetime.timedelta(days=13)
for cde in dict_of_lists:
    if cde == beginner:
        not_complete_days_dict[cde] = dict_of_lists[cde]
        beginner += datetime.timedelta(days=1)

tid = 15
fraction_list = []
true_inc_dict = {}
dates_list3 = [*not_complete_days_dict]
dates_list3.sort()
# dates_list3 = dates_list3[::-1]
true_list=[]
ttrue_list=[]


for nc in dates_list3:
    next_fract = (true_real_fract_dict_sum[tid]/dict_sum_days_whereXisreal[tid])*.9
    fraction_list.append(next_fract)
    sum_fl = sum(fraction_list)
    true_total = int(dict_sum_by_date[nc]*(1/(1-sum_fl)))
    true_inc_dict[nc] = true_total
    ttrue_list.append(true_total)
    for gr in range(1, tid,1):
        dict_sum_days_whereXisreal[gr] += true_total
    for grain in range(1, tid,1):
        true_real_fract_dict_sum[grain] += dict_of_lists[nc][grain - 1]
    tid-=1

first = (true_real_fract_dict_sum[1]/dict_sum_days_whereXisreal[1])
fraction_list.insert(len(fraction_list), first)
norm_inter = 1/(sum(fraction_list))
fraction_list = [i * norm_inter for i in fraction_list]


tdtty_list = [*dict_of_lists]
tdtty_list.sort()
true_tot_dict = {}
for tdat in tdtty_list:
    if tdat < ch:
        true_tot_dict[tdat] = dict_sum_by_date[tdat]
    if tdat >= ch:
        if tdat < very_end_date_1 - datetime.timedelta(days=13):
            true_tot_dict[tdat] = middle_dict[tdat]
        if tdat >= very_end_date_1- datetime.timedelta(days=13):
            true_tot_dict[tdat] = true_inc_dict[tdat]

tll = [*true_tot_dict]
tll.sort()
for tlll in tll:
    true_list.append(true_tot_dict[tlll])
stt = int(sum(true_list))


xnp = range(len(true_list))
aye = range(100)
ynp = true_list
znp = np.polyfit(xnp, ynp, 4)
ev_lobfit = polyval(znp, aye)
list_ev_lobfit = np.array(ev_lobfit).tolist()
llist_ev_lobfit = list_ev_lobfit[:70]
# mean = llist_ev_lobfit.index(max(llist_ev_lobfit))
mean = 46

dist = 0
diff_list = []
for daily in true_list:
    front_diff = daily * ((dist-mean)*(dist-mean))
    diff_list.append(front_diff)
    dist+=1


int_sum = [int(i) for i in diff_list]
sum_diff = sum(diff_list[0:mean])
both_sides = sum_diff*2
int_ft = int(stt)
sigma = math.sqrt((both_sides/int_ft))
mu = mean
# need to split r hand of normal curve in sections between latest value and 3.4 sd
# total area is l/h - total true values after mean. slices = mean-no of days recorded after mean.
# z

total_slices = mean*2
slices_left = total_slices-len(true_list)

z_scores = []
fst = 3.4
z_scores.append(fst)#
gr = 3.4/mean
inc = 0
for z in range (slices_left):
    score = z_scores[z] - gr
    z_scores.append(score)
p_val = stats.norm.sf(np.abs(z_scores))
ah = np.array(p_val).tolist()
array_length = len(p_val)
last_element = p_val[array_length- 1]
helper = 1/last_element
p_values=[]
for p in p_val:
    q = p*helper
    p_values.append(q)
big_totals = []
up_to_now = true_list
total_ex_toll = sum(up_to_now)*(1/(1-last_element))
toll_left = total_ex_toll-sum(up_to_now)


for pval in p_values:
    bt= pval * toll_left
    big_totals.append(int(bt))


subbed = np.diff(big_totals)
list_sub = np.array(subbed).tolist()
rev_list_sub = list_sub[::-1]
rev_list_sub.append(3)

# make announced predictions
rev_frac = fraction_list[::-1]

# make full dict
true_full_list = []
for tf in true_list:
    true_full_list.append(tf)
for fer in rev_list_sub:
    true_full_list.append(fer)
for ra in range(20):
    true_full_list.append(0)


true_full_dict = {}
dx = datetime.date(2020, 3, 1)

for date in true_full_list:
    true_full_dict[dx] = date
    dx += datetime.timedelta(days=1)


announce_total = []
range_finder_s = len(true_list)
range_finder_e = len(true_full_list)

for dayday in range(range_finder_s, range_finder_e):
    sub_list = []
    end_range = dayday-20
    day_reducer = dayday
    rf = 1
    for dd in range (13, -1, -1):
        to_add = true_full_list[day_reducer-2]*rev_frac[rf]
        sub_list.append(to_add)
        day_reducer-=1
        rf+=1
    sumsum = sum(sub_list)
    announce_total.append(sumsum)
announced_total = [int(w) for w in announce_total]

edn = datetime.date(2020, 3, 1)
dict_true = {}
for each_day in true_list:
    dict_true[edn] = each_day
    edn += datetime.timedelta(days=1)

pred_dict = {}

today = datetime.date.today()
dvx = today
dvx -= datetime.timedelta(days=1)
if dvx in dict_true:
    tdtest=0
else:
    tdtest=1

dt = today
for dat in announced_total:
    if tdtest == 0:
        dt += datetime.timedelta(days=1)
    pred_dict[dt] = int(dat/(1+ons_multiplier))
    dt += datetime.timedelta(days=1)

def bar_graph(*args):
    y_pos = np.arange(len(args))+1
    plt.bar(y_pos, args,  align='center', alpha=0.5)
    plt.show()

def bar_graph_2(**kwargs):
    lists = sorted(kwargs.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.bar(x, y, align='center', alpha=0.5)
    plt.xticks(rotation=90)
    plt.show()

def curve_graph(**kwargs):
    lists = sorted(kwargs.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.bar(x, y, align='center', alpha=0.5)
    plt.plot(xnp, polyval(znp, xnp))
    plt.xticks(rotation=90)
    plt.ylim(bottom=0)
    plt.show()



inp = input("Choose your stat\n"
            "a for total toll announced to date\n"
            "b for true total to date\n"
            "c for predicted toll\n"
            "d for true toll on certain date, past or future\n"
            "e for predicted announced toll for certain date\n"
            "f for graph of fractions\n"
            "g for graph of true toll to date\n"
            "h for graph of predicted full toll\n"
            "i for graph of predicted announced toll\n"
            "j for graph of curve\n"
            "k for percentage underestimated by daily toll\n"
            "l for true sum up to and including a certain date\n")

if inp == 'a':
    print(full_total)

if inp == 'b':
    print(stt)

if inp == 'c':
    print(int(sum(true_full_list)))

if inp == 'd':
    mn = input('Enter month(eg for April, enter 4)\n')
    dt = input('Enter date\n')
    imn = int(mn)
    idt = int(dt)
    da = datetime.date(2020, imn, idt)
    ttt=0
    if da not in dict_true:
        ttt += 1
    for k in  true_full_dict:
        if k == da:
            print(true_full_dict[k])
            if ttt ==1:
                print(' - prediction')
            else:
                print(' - lag added to announced toll')


if inp == 'e':
    mnb = input('Enter month(eg for April, enter 4)\n')
    dtb = input('Enter date\n')
    imnb = int(mnb)
    idtb = int(dtb)
    dbb = datetime.date(2020, imnb, idtb)
    if dbb not in pred_dict:
        print("date either already announced or too far in future")
    for n in pred_dict:
        if n == dbb:
            print(pred_dict[n])

if inp == 'f':
    bar_graph(*rev_frac)

if inp == 'g':
    dict_true = {str(k): v for k, v in dict_true.items()}
    curve_graph(**dict_true)

if inp == 'h':
    true_full_dict = {str(k): v for k, v in true_full_dict.items()}
    bar_graph_2(**true_full_dict)

if inp == 'i':
    pred_dict = {str(k): v for k, v in pred_dict.items()}
    bar_graph_2(**pred_dict)

if inp == 'j':
    true_full_dict = {str(k): v for k, v in true_full_dict.items()}
    curve_graph(**true_full_dict)


if inp == 'k':
    a = stt-full_total
    perc = (100/full_total)*a
    print(int(perc))

if inp == 'l':
    hold_list = []
    mnbs = input('Enter month(eg for April, enter 4)\n')
    dtbs = input('Enter date\n')
    imnbs = int(mnbs)
    idtbs = int(dtbs)
    dbbs = datetime.date(2020, imnbs, idtbs)
    for ev in dict_true:
        if ev <= dbbs:
            hold_list.append(dict_true[ev])
    if dbbs not in dict_true:
        print("no sum for that date")
    else:
        integ_sum = [int(i) for i in hold_list]
        summer = int(sum(integ_sum))
        print(summer)