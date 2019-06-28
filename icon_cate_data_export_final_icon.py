f_icon = open('icon_fold0_testset.txt', 'r')
f_ss = open('sc_fold0_testset.txt', 'r')
f_output = open('icon_spreadsheet_final.txt', 'w')
f_output.close()
f_output = open('icon_spreadsheet_final.txt', 'a')

print_app_id = False

#make icon dict
icon_dict = {}
for line in f_icon:
    s = line.split(' ')
    s[-1] = s[-1][:-1]
    if s[1] != '':
        icon_dict[s[0]] = int(s[1])
    else:
        icon_dict[s[0]] = -1

#write app_id by screenshot
for line in f_ss:
    app_id = line.split(' ')[0]
    if app_id in icon_dict:
        if print_app_id:
            f_output.write(app_id + ' ' + str(icon_dict[app_id]) + '\n')
        else:
            f_output.write(str(icon_dict[app_id]) + '\n')

        del icon_dict[app_id]
    else:
        f_output.write(app_id + ' -1\n')
        del icon_dict[app_id]
#add left by app_id of icon
f_output.write('##################################\n')
for k,v in icon_dict.items():
    if print_app_id:
        f_output.write(k + ' ' + str(v) +'\n')
    else:
        f_output.write(str(v) +'\n')
f_output.close()