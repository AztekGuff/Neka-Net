from tkinter import *

window = Tk()
window.title('i want to break free')
window.resizable(False,False)

b1 = Button(window,text='fist',height=2,width=10).grid(column=1,row=1,columnspan=1,rowspan=1)
b2 = Button(window,text='ass', height=2,width=10).grid(column=1,row=2,columnspan=1,rowspan=1)

str=StringVar()
text = Label(window,textvariable=str,anchor='nw',
             font='Helvetica 11',justify=LEFT,
             fg='SpringGreen2',bg='black',
             height=35,width=125).grid(column=2,row=1,columnspan=10,rowspan=100)

str.set('asdasdasaaaaaaaaaaaaadas\nasdddddddd')



window.mainloop()
