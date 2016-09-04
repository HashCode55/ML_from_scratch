from Tkinter import *
import tkMessageBox

COMPANIES = [
	'SBI', 
	'AIR INDIA',
	'VODAFONE',
	'PETROCHINA',
	'HTC'
	]
COMPANY_PRICES = {
	'SBI':100.0, 
	'AIR INDIA':100.0,
	'VODAFONE':100.0,
	'PETROCHINA':100.0,
	'HTC':100.0
	}	
COMPANY_VOLUMES = {
	'SBI':6000.0, 
	'AIR INDIA':6000.0,
	'VODAFONE':6000.0,
	'PETROCHINA':6000.0,
	'HTC':6000.0
	}	
def which_selected():
	return int(select.curselection()[0])

banner_new = -1
banner_old = -1
def update(company, new_price, new_vol, label1, label2, label3):
	#initial variables 
	ini_price = COMPANY_PRICES[company]
	ini_volume = COMPANY_VOLUMES[company]
	ini_total_val = ini_price*ini_volume
	#new_price = new_price
	#new_vol = new_vol
	#calculate the new price 
	new_total = new_price*new_vol
	dp = new_price - ini_price 
	fv = new_total/ini_total_val
	mp = (new_price+ini_price)/2.0
	fdp = dp/mp
	fdv = new_vol/ini_volume
	fdv1 = fdv + 0.5
	xp = fdp*fv*fdv1
	new_fin_price = (1.0+xp)*ini_price
	COMPANY_PRICES[company] = new_fin_price
	banner_old = ini_price
	banner_new = new_fin_price	
	label1.config(text=company)
	label2.config(text=banner_old)
	label3.config(text=banner_new)

def make_window():
	win = Tk()

	frame1 = Frame(win, pady = 20)
	frame1.pack()

	Label(frame1, text = "Stock Price").grid(row = 0,  column = 0, sticky = W)
	stk_price = DoubleVar()
	name = Entry(frame1, textvariable = stk_price)
	name.grid(row = 0, column = 1, sticky = W)

	Label(frame1, text="Stock Volume").grid(row=1, column=0, sticky=W)
    	stk_volume = DoubleVar()
    	phone= Entry(frame1, textvariable=stk_volume)
    	phone.grid(row=1, column=1, sticky=W)

    	#Create the dropdown menu
    	companies_option = StringVar(win)
	#set the default value 
	companies_option.set(COMPANIES[0])
	w = apply(OptionMenu, (win, companies_option) + tuple(COMPANIES))
	w.pack()

    	
    	frame2 = Frame(win, pady =10)
    	frame2.pack()
    	#if the update has been used update the banner 
    	l1 = Label(frame2, text=companies_option.get())
    	l1.grid(row=1, column=0, sticky=W)
    	l2 = Label(frame2, text="Old Price: "+str(banner_old))
    	l2.grid(row=2, column=0, sticky=W)
    	l3 = Label(frame2, text="New Price: " + str(banner_new))
    	l3.grid(row=3, column=0, sticky=W)
    	frame2 = Frame(win, pady = 20)
    	frame2.pack()
    	#update the stock shit
    	b1 = Button(frame2,text=" Update Company Value", command = lambda : update(
    		companies_option.get(),
    		 stk_price.get(),
    		 stk_volume.get(),
    		 l1, 
    		 l2, 
    		 l3))
    	b1.pack(side=LEFT)
    	return win

win = make_window()
win.geometry("500x300")
win.title("Stock Market Simulator")
win.mainloop()

