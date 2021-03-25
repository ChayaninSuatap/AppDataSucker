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

from shutil import copyfile
import os     
import sc_util

def prepare_icons(icon_path, sc_path, icon_output_path, sc_output_path, type_d, sc_dict):
    for type, app_ids in type_d.items():
        if not os.path.exists(icon_output_path + '/' + type):
            os.mkdir(icon_output_path + '/' + type)
        if not os.path.exists(sc_output_path + '/' + type):
            os.mkdir(sc_output_path + '/' + type)
        
        for app_id in app_ids:
            #icon
            copyfile(icon_path + '/' + app_id + '.png', icon_output_path + '/' + type + '/' + app_id + '.png')

            #screenshot
            for sc_fn in sc_dict[app_id]:
                copyfile(sc_path + '/' + sc_fn, sc_output_path + '/' + type + '/' + sc_fn)


if __name__ == '__main__':
    sc_dict = sc_util.make_sc_dict('screenshots.256.distincted.rem.human/')
    prepare_icons(
        icon_path='icons_rem_dup_human_recrawl/icons_rem_dup_recrawl',
        sc_path='screenshots.256.distincted.rem.human/',
        icon_output_path='journal/sim_search/sports/icons',
        sc_output_path='journal/sim_search/sports/screenshots',
        type_d=sports,
        sc_dict=sc_dict)
    # prepare_screenshots()
