import Foundation
import Darwin

private let usage = """
Uso:
  LSMExtractorCLI --input <dir> --output <dir> [opciones]

Extrae landmarks de manos y pose (Apple Vision) desde videos
organizados como {input}/{category}/{word}/*.{mp4,mov,m4v}.

El JSON de salida usa el mismo esquema que LSMExtractorGUI
(frame, timestamp_ms, leftHand, rightHand, pose; Y=0 arriba).

Opciones:
  -i, --input <dir>       Carpeta raíz del dataset (obligatorio)
  -o, --output <dir>      Carpeta de salida (obligatorio)
  -m, --manifest <file>   dataset_manifest.json existente a cargar y actualizar
      --overlay           Genera {base}_overlay.mp4 (por defecto solo JSON)
      --skip-existing     Omite videos cuyo *_landmarks.json ya existe
  -h, --help              Muestra esta ayuda

Códigos de salida:
  0  lote terminado sin errores
  1  argumentos o rutas inválidos, o no se pudo guardar manifiesto/reporte
  2  al menos un video falló
"""

struct CLIError: Error {
    let message: String
}

enum CLIMain {

    static func run(arguments: [String]) async -> Int32 {
        if arguments.contains(where: { $0 == "--help" || $0 == "-h" }) {
            print(usage)
            return 0
        }
        if arguments.isEmpty {
            fputs(usage + "\n", stderr)
            return 1
        }

        do {
            let options = try parse(arguments)
            try validate(options)
            return await BatchRunner().run(options)
        } catch let error as CLIError {
            fputs("\(error.message)\n\n\(usage)\n", stderr)
            return 1
        } catch {
            fputs("\(error.localizedDescription)\n", stderr)
            return 1
        }
    }

    private static func parse(_ args: [String]) throws -> CLIOptions {
        var input: String?
        var output: String?
        var manifest: String?
        var overlay = false
        var skipExisting = false
        var index = 0

        while index < args.count {
            let arg = args[index]
            switch arg {
            case "--input", "-i":
                input = try value(after: &index, in: args, flag: arg)
            case "--output", "-o":
                output = try value(after: &index, in: args, flag: arg)
            case "--manifest", "-m":
                manifest = try value(after: &index, in: args, flag: arg)
            case "--overlay":
                overlay = true
                index += 1
            case "--skip-existing":
                skipExisting = true
                index += 1
            default:
                throw CLIError(message: "Opción desconocida: \(arg)")
            }
        }

        guard let input else { throw CLIError(message: "Falta --input") }
        guard let output else { throw CLIError(message: "Falta --output") }

        return CLIOptions(
            inputURL: resolve(input),
            outputURL: resolve(output),
            manifestURL: manifest.map(resolve),
            generateOverlay: overlay,
            skipExisting: skipExisting
        )
    }

    private static func value(after index: inout Int, in args: [String], flag: String) throws -> String {
        let next = index + 1
        guard next < args.count, !args[next].hasPrefix("-") else {
            throw CLIError(message: "Falta el valor de \(flag)")
        }
        index = next + 1
        return args[next]
    }

    private static func resolve(_ raw: String) -> URL {
        let expanded = (raw as NSString).expandingTildeInPath
        let absolute: String
        if (expanded as NSString).isAbsolutePath {
            absolute = expanded
        } else {
            absolute = (FileManager.default.currentDirectoryPath as NSString).appendingPathComponent(expanded)
        }
        // realpath, not URL.resolvingSymlinksInPath: the latter leaves /tmp
        // unresolved while directory enumeration returns /private/tmp.
        return URL(fileURLWithPath: canonicalPath(absolute))
    }

    private static func canonicalPath(_ path: String) -> String {
        if let resolved = realpath(path, nil) {
            defer { free(resolved) }
            return String(cString: resolved)
        }
        let ns = path as NSString
        let parent = ns.deletingLastPathComponent
        if parent == path || parent == "/" || parent.isEmpty {
            return path
        }
        return (canonicalPath(parent) as NSString).appendingPathComponent(ns.lastPathComponent)
    }

    private static func validate(_ options: CLIOptions) throws {
        let fm = FileManager.default
        var isDir: ObjCBool = false

        guard fm.fileExists(atPath: options.inputURL.path, isDirectory: &isDir), isDir.boolValue else {
            throw CLIError(message: "La carpeta de entrada no existe: \(options.inputURL.path)")
        }

        if fm.fileExists(atPath: options.outputURL.path, isDirectory: &isDir), !isDir.boolValue {
            throw CLIError(message: "La ruta de salida no es una carpeta: \(options.outputURL.path)")
        }

        if let manifestURL = options.manifestURL {
            guard fm.fileExists(atPath: manifestURL.path, isDirectory: &isDir), !isDir.boolValue else {
                throw CLIError(message: "El manifiesto no existe: \(manifestURL.path)")
            }
        }
    }
}

let exitCode = await CLIMain.run(arguments: Array(CommandLine.arguments.dropFirst()))
exit(exitCode)
