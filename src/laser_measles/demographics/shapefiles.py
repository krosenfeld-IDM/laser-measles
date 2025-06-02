from pathlib import Path
from shapefile import Reader, Writer
import alive_progress

def check_dotname(path: str | Path):
    path = Path(path) if isinstance(path, str) else path
    with Reader(path) as sf:
        fields = [field[0] for field in sf.fields[1:]]
        if 'DOTNAME' in fields:
            return True
        return False

def add_dotname(
    path: str | Path,
    verbose: bool = True,
    dot_name_fields: list[str] = ["COUNTRY","NAME_1", "NAME_2"],
    dotname_symbol: str = ":",
    new_name: str = "new",
    inplace: bool = False,
) -> None:
    """
    Add a DOTNAME to the shapefile.
    """

    def make_temp_path(path: Path, suffix: str) -> Path:
        return path.with_name(path.stem + "_" + new_name + suffix)

    # Resolve shapefile
    path = Path(path) if isinstance(path, str) else path

    # original shapefile
    with Reader(path) as sf:

        fields = [field[0] for field in sf.fields[1:]]
        if not all(field in fields for field in dot_name_fields):
            raise ValueError(
                f"Dot name fields {dot_name_fields} not found in shapefile {path}. Choices are {fields}"
            )
        
        if 'DOTNAME' in fields:
            return

        dotnames = [dotname_symbol.join([shaperec.record[field].lower() for field in dot_name_fields]) for shaperec in sf.iterShapeRecords()]

        # check that all dotnames are unique
        if len(dotnames) != len(set(dotnames)):
            raise ValueError(f"Dotnames are not unique in shapefile {path}")


        # create a new shapefile
        with Writer(make_temp_path(path, path.suffix)) as w:

            # add the original fields
            for i, field in enumerate(sf.fields):
                if i > 0:
                    w.field(*field)
            # add the new field
            w.field('DOTNAME', 'C', 50)

            record_cnt = 0

            for i, shaperec in enumerate(sf.iterShapeRecords()):
                dotname = dotname_symbol.join(
                    [shaperec.record[field].lower() for field in dot_name_fields]
                )
                # add the new field
                w.record(*shaperec.record, dotname)
                # add the shape
                w.shape(shaperec.shape)
                record_cnt += 1

    # copy the new shapefile to the old
    if inplace:
        for suffix in [".shp", ".shx", ".prj", ".cpg", ".prj", ".dbf"]:
            temp_path = make_temp_path(path, suffix)
            if temp_path.exists():
                temp_path.rename(path.with_suffix(suffix))
        
