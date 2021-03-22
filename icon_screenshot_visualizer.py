from global_util import load_pickle
icon_path = 'D:/crawler/icons_rem_dup_human_recrawl/icons_rem_dup_recrawl'
sc_path = 'C:/screenshots.resized'
output_path = 'journal/icon_screenshot_visualizer'
import sc_util
sc_dict = sc_util.make_sc_dict(sc_path)
from shutil import copyfile

cates = ['BOARD', 'TRIVIA',	'ARCADE','CARD','MUSIC','RACING','ACTION','PUZZLE','SIMULATION','STRATEGY','ROLE_PLAYING','SPORTS','ADVENTURE','CASINO','WORD','CASUAL','EDUCATIONAL']

template_header = '<html><body><table border=1px><tr><th>icon</th>'
for i in range(20):
    template_header += '<th> screenshot %d</th>' % (i+1,)
template_header += '</tr>'
template_footer = '</table></body></html>'

# aial = load_pickle('aial_seed_327.obj')
# with open(output_path + '/index.html', 'w') as f:
#     f.write(template_header)
#     for app_id, _ , cate , *_ in aial:
#         if cates[cate.index(1)] == 'SPORTS':
#             f.write('<tr>')
#             f.write('<td><div style="max-width:180px;word-wrap:break-word;">%s</div><div><img width=180 height=180 src="%s"></img></div></td>' % (app_id, icon_path + '/' + app_id + '.png'))

#             #screenshot
#             if app_id in sc_dict:
#                 for sc_fn in sc_dict[app_id]:
#                     f.write('<td><div></div><div><img src="%s"></img></div></td>' % (sc_path + '/' + sc_fn,))

#             f.write('</tr>')
#     f.write(template_footer)

sports = {
    'football' : ['com.ahoiii.FieteSoccer', 'com.batovi.pixelcupsoccer2', 'com.cg.football', 'com.djinnworks.ss16', 'com.djinnworks.ss18' ,'com.djinnworks.StickmanSoccer','com.djinnworks.StickmanSoccer2014','com.ea.gp.fifamobile','com.ea.gp.fifaworld','com.gameloft.android.ANMP.GloftR7HM','com.magiccubegames.dumberleague','com.touchtao.soccerkinggoogle','com.turner.tooncup','com.uplayonline.strikersocceramerica','com.uplayonline.strikersoccerbrasil','jp.konami.pesam','kz.snailwhale.soccer'],
    'bowling' : ['bowling.master.club.free.android', 'com.concretesoftware.pbachallenge_androidmarket', 'com.eightsec.StrikyBall', 'com.eivaagames.Bowling3DExtreme', 'com.indptechnologiesbowing', 'com.magmamobile.game.Bowling3D.XMas', 'com.mobirix.bowling', 'com.pnixgames.bowlingking', 'com.three.dimension.bowling.battle.free', 'com.threed.bowling'],
    'fishing' : ['com.bigsportfishing.bsf2', 'com.com2us.acefishing.normal.freefull.google.global.android.common', 'com.concretesoftware.rapala', 'com.inertiasoftware.fishingworld', 'com.jzb.fishingmania', 'com.mobirix.fishingchampionship', 'com.obul.realfishing', 'com.rocketmind.fishingfull', 'com.rocketmind.riverfishing', 'com.vetti.realfishingace'],
    'boxing' : ['com.appsolute.pixelpunchers', 'com.boxingclub.realpunch.free', 'com.ea.game.easportsufc_row', 'com.vividgames.realboxing', 'com.yx.boxinghero'],
    'snooker': ['board.inter.pool', 'com.andromedagames.pool2018free', 'com.arcadegame.games.ball.pool.billiards', 'com.billiards.city.pool.nation.club', 'com.celeris.VirtualPool', 'com.crossfield.poolmaster', 'com.eivaagames.PoolBilliards3D', 'com.forthblue.pool', 'com.gamedesire.snookerlivepro', 'com.giraffegames.pool', 'com.miniclip.eightballpool', 'com.mobirix.pocket8ball', 'com.ticktockgames.isnookerpro', 'com.tigereyegames.wc3c', 'com.uken.pool', 'com.webzen.pocket.google', 'com.xs.pooltd', 'eu.friendlymonster.totalsnookerclassic', 'game.snooker.billiard.pool', 'jp.co.arttec.satbox.Billiards9', 'thai.pool.thai', 'uk.co.bigheadgames.Snooker'],
    'tennis' : ['com.cg.tennis', 'com.djinnworks.StickmanTennis2015', 'com.jakyl.tcr', 'com.ninemgames.tennis.google', 'com.ninemgames.tennis2.google', 'com.tennisleague.pocketgames', 'net.kairosoft.android.tennis_ja'],
    'golf' : ['com.com2us.golfstarworldtour.normal.freefull.google.global.android.common', 'com.eivaagames.MiniGolfGame3D', 'com.fullfat.android.flickgolf', 'com.playdemic.golf.android', 'com.spacegame.golf.ace', 'com.webzen.shotonlinegolf.google', 'com.wordsmobile.golfchampionship'],
    'baseball' : ['com.com2us.ninepb3d.normal.freefull.google.global.android.common', 'com.com2us.probaseball3d.normal.freefull.google.global.android.common', 'com.doodlemobile.realbaseball', 'com.fullfat.blockybaseball', 'com.gamevilusa.mlbpilive.android.google.global.normal', 'com.mlb.HomeRunDerby', 'com.mlb.RBIBaseball2018',  'us.kr.baseballnine'],
    'pingpong' : ['com.crossfield.tabletennis3d', 'com.giraffegames.tabletennis', 'com.giraffegames.tabletennis3d', 'com.orangenose.tablefull', 'uk.co.yakuto.TableTennisTouch'],
    'basketball' : ['com.t2ksports.nba2k19and', 'com.ea.gp.nbamobile', 'com.hotheadgames.google.free.bigwinbasketball', 'com.triples.game.freestyle.tha', 'com.djinnworks.sb17', 'com.wanxing.basketball', 'com.YOON.DOUBLECLUTCH'],
    'cricket' : ['Ashes2010_androidmkp.indvseng', 'com.appon.worldofcricket', 'com.games2win.worldcupcricketchamp', 'com.indvspak.ashes', 'com.ipl.t20pl2015', 'com.jetplay.sachinsagacc', 'com.moonfrog.cricket', 'com.moonglabs.epiccricket', 'com.nautilus.RealCricketTestMatchEdition', 'com.nazara.viratsupercricket', 'com.nextwave.bigbash', 'com.nextwave.wcc_lt', 'com.renderedideas.cricket', 'com.zapak.gl2017', 'com.zapak.worldcup.t20.cricket', 'IndVsAus2012_androidmkp.indvsaus', 'org.cocos2dx.NautilusCricket2014', 't20cricket2012_androidmkp.extraaa_innings_t20']
}

#copy images
# for subcate, app_ids in sports.items():
#     for app_id in app_ids:
#         copyfile(icon_path + '/' + app_id + '.png', 'journal/icon_screenshot_visualizer/sports/icons/' + app_id + '.png')
#         for sc_fn in sc_dict[app_id]:
#             copyfile(sc_path + '/' + sc_fn, 'journal/icon_screenshot_visualizer/sports/screenshots/' + sc_fn)

for subcate, app_ids in sports.items():
    with open('%s/sports/%s.html' % (output_path, subcate), 'w') as f:
        for app_id in app_ids:
            f.write(template_header)
            f.write('<tr>')
            f.write('<td><div style="max-width:180px;word-wrap:break-word;">%s</div><div><img width=180 height=180 src="%s"></img></div></td>' % (app_id, 'icons/' + app_id + '.png'))
            #screenshot
            for sc_fn in sc_dict[app_id]:
                f.write('<td><div></div><div><img src="%s"></img></div></td>' % ('screenshots/' + sc_fn,))
            f.write('</tr>')
        f.write(template_footer)




