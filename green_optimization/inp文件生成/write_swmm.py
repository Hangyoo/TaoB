#coding=utf-8
__author__ = 'lly'
with open ('pjxc_blank.inp') as f:
    lines=f.readlines()##文件内容读取列表

    a=open('SUBCATCHMENTS.txt')
    subcatchment=a.readlines()##文件内容读取列表
    subcatchment=" ".join(subcatchment)
    for line in lines:
        if line=='[SUBCATCHMENTS]\n':
            num_1=lines.index(line)
            lines.insert((num_1+3),subcatchment)

    b=open('SUBAREA.txt')
    subareas=b.readlines()##文件内容读取列表
    subareas=" ".join(subareas)
    for line in lines:
        if line=='[SUBAREAS]\n':
            num_2=lines.index(line)
            lines.insert((num_2+3),subareas)

    c=open('INFILTRATION.txt')
    infiltration=c.readlines()##文件内容读取列表
    infiltration=" ".join(infiltration)
    for line in lines:
        if line=='[INFILTRATION]\n':
            num_3=lines.index(line)
            lines.insert((num_3+3),infiltration)

    d=open('LID_USAGE.txt')
    lid_usage=d.readlines()##文件内容读取列表
    lid_usage=" ".join(lid_usage)
    for line in lines:
        if line=='[LID_USAGE]\n':
            num_4=lines.index(line)
            lines.insert((num_4+3),lid_usage)

    e=open('JUNCTIONS.txt')
    junction=e.readlines()##文件内容读取列表
    junction=" ".join(junction)
    for line in lines:
        if line=='[JUNCTIONS]\n':
            num_5=lines.index(line)
            lines.insert((num_5+3),junction)

    f=open('OUTFALLS.txt')
    outfall=f.readlines()##文件内容读取列表
    outfall=" ".join(outfall)
    for line in lines:
        if line=='[OUTFALLS]\n':
            num_6=lines.index(line)
            lines.insert((num_6+3),outfall)

    i=open('CONDUITS.txt')
    couduit=i.readlines()##文件内容读取列表
    couduit=" ".join(couduit)
    for line in lines:
        if line=='[CONDUITS]\n':
            num_7=lines.index(line)
            lines.insert((num_7+3),couduit)

    j=open('XSECTIONS.txt')
    xsection=j.readlines()##文件内容读取列表
    xsection=" ".join(xsection)
    for line in lines:
        if line=='[XSECTIONS]\n':
            num_8=lines.index(line)
            lines.insert((num_8+3),xsection)

    k=open('COVERAGE.txt')
    coverage=k.readlines()##文件内容读取列表
    coverage=" ".join(coverage)
    for line in lines:
        if line=='[COVERAGES]\n':
            num_9=lines.index(line)
            lines.insert((num_9+3),coverage)

    l=open('COORDINATES.txt')
    COORDINATES=l.readlines()##文件内容读取列表
    COORDINATES=" ".join(COORDINATES)
    for line in lines:
        if line=='[COORDINATES]\n':
            num_10=lines.index(line)
            lines.insert((num_10+3),COORDINATES)

    m=open('Polygons.txt')
    Polygons=m.readlines()##文件内容读取列表
    Polygons=" ".join(Polygons)
    for line in lines:
        if line=='[Polygons]\n':
            num_11=lines.index(line)
            lines.insert((num_11+3),Polygons)


with open ('sz_pjxc_for_efdc.inp','w',encoding='utf-8') as f:
    for line in lines:
        f.write(line)

